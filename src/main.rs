#![cfg_attr(coverage_nightly, feature(coverage_attribute))]

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::io::{Write, stderr, stdout};
use std::time::{Duration, Instant};

const API_BASE: &str = "https://openrouter.ai/api/v1";

#[derive(Parser)]
#[command(name = "openrouter", about = "OpenRouter API CLI")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// List available models
    Models {
        #[arg(short, long)]
        category: Option<String>,
        #[arg(short, long)]
        search: Option<String>,
        #[arg(short, long)]
        pricing: bool,
    },
    /// Send a chat completion request
    Query {
        /// Model ID (e.g. deepseek/deepseek-v4-flash)
        model: String,
        /// Prompt text
        prompt: String,
        /// System prompt
        #[arg(short, long)]
        system: Option<String>,
        /// Maximum tokens to generate
        #[arg(long, default_value_t = 256)]
        max_tokens: u32,
        /// Disable streaming
        #[arg(long)]
        no_stream: bool,
    },
}

#[derive(Debug, Deserialize)]
struct ModelsResponse {
    data: Vec<Model>,
}

#[derive(Debug, Deserialize)]
struct Model {
    id: String,
    name: String,
    context_length: Option<u64>,
    pricing: Option<Pricing>,
}

#[derive(Debug, Deserialize)]
struct Pricing {
    prompt: String,
    completion: String,
}

#[derive(Debug, PartialEq, Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<Message>,
    max_tokens: u32,
    stream: bool,
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
struct Message {
    role: String,
    #[serde(default)]
    content: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ChatResponse {
    choices: Vec<Choice>,
    usage: Option<Usage>,
}

#[derive(Debug, Deserialize)]
struct Choice {
    message: Message,
}

#[derive(Debug, Deserialize)]
struct Usage {
    prompt_tokens: u64,
    completion_tokens: u64,
}

#[derive(Debug, Default, PartialEq)]
struct StreamStats {
    prompt_tokens: u64,
    completion_tokens: u64,
}

#[tokio::main]
#[cfg_attr(coverage_nightly, coverage(off))]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Models {
            category,
            search,
            pricing,
        } => list_models(category, search, pricing).await?,
        Commands::Query {
            model,
            prompt,
            system,
            max_tokens,
            no_stream,
        } => query(model, prompt, system, max_tokens, !no_stream).await?,
    }

    Ok(())
}

// ── Models ──────────────────────────────────────────────────────────────────

#[cfg_attr(coverage_nightly, coverage(off))]
async fn list_models(
    category: Option<String>,
    search: Option<String>,
    show_pricing: bool,
) -> Result<()> {
    let client = reqwest::Client::new();
    let url = build_models_url(category.as_deref());

    let resp: ModelsResponse = client
        .get(&url)
        .send()
        .await
        .context("Failed to fetch models")?
        .json()
        .await
        .context("Failed to parse response")?;

    let models = filter_models(resp.data, search.as_deref());

    println!("Found {} models\n", models.len());

    for model in &models {
        print_model_line(model, show_pricing);
    }

    Ok(())
}

fn build_models_url(category: Option<&str>) -> String {
    let mut url = format!("{}/models", API_BASE);
    if let Some(cat) = category {
        url = format!("{}?category={}", url, cat);
    }
    url
}

fn filter_models(models: Vec<Model>, search: Option<&str>) -> Vec<Model> {
    let Some(term) = search else {
        return models;
    };
    let term = term.to_lowercase();
    models
        .into_iter()
        .filter(|m| {
            let name = m.name.to_lowercase();
            let id = m.id.to_lowercase();
            name.contains(&term) || id.contains(&term)
        })
        .collect()
}

#[cfg_attr(coverage_nightly, coverage(off))]
fn print_model_line(model: &Model, show_pricing: bool) {
    print!("{:50}", model.id);
    if let Some(ctx) = model.context_length {
        print!("  {:>7}k ctx", ctx / 1000);
    } else {
        print!("  {:>10}", "");
    }
    if show_pricing && let Some(pricing) = &model.pricing {
        print!(
            "  {:>8}/M in  {:>8}/M out",
            format_price(&pricing.prompt),
            format_price(&pricing.completion)
        );
    }
    println!();
}

fn format_price(price_str: &str) -> String {
    let price: f64 = price_str.parse().unwrap_or(0.0);
    let per_million = price * 1_000_000.0;
    format!("${:.2}", per_million)
}

// ── Query ───────────────────────────────────────────────────────────────────

#[cfg_attr(coverage_nightly, coverage(off))]
async fn query(
    model: String,
    prompt: String,
    system: Option<String>,
    max_tokens: u32,
    do_stream: bool,
) -> Result<()> {
    let api_key =
        std::env::var("OPENROUTER_API_KEY").context("OPENROUTER_API_KEY env var not set")?;
    let body = build_chat_request(model.clone(), prompt, system, max_tokens, do_stream);

    let client = reqwest::Client::new();
    let url = format!("{}/chat/completions", API_BASE);
    let start = Instant::now();

    if do_stream {
        query_streaming(client, url, api_key, body, start, &model).await
    } else {
        query_blocking(client, url, api_key, body, start, &model).await
    }
}

fn build_chat_request(
    model: String,
    prompt: String,
    system: Option<String>,
    max_tokens: u32,
    stream: bool,
) -> ChatRequest {
    let mut messages = Vec::new();
    if let Some(sys) = system {
        messages.push(Message {
            role: "system".into(),
            content: Some(sys),
        });
    }
    messages.push(Message {
        role: "user".into(),
        content: Some(prompt),
    });

    ChatRequest {
        model,
        messages,
        max_tokens,
        stream,
    }
}

#[cfg_attr(coverage_nightly, coverage(off))]
async fn query_streaming(
    client: reqwest::Client,
    url: String,
    api_key: String,
    body: ChatRequest,
    start: Instant,
    model: &str,
) -> Result<()> {
    let response = client
        .post(&url)
        .bearer_auth(&api_key)
        .json(&body)
        .send()
        .await
        .context("Failed to send request")?;

    if !response.status().is_success() {
        let status = response.status();
        let text = response.text().await.unwrap_or_default();
        bail!("API error {}: {}", status, text);
    }

    let mut byte_stream = response.bytes_stream();
    let mut ttft: Option<std::time::Duration> = None;
    let mut stats = StreamStats::default();
    let mut buf = String::new();
    let mut stdout = stdout();

    while let Some(chunk) = byte_stream.next().await {
        let chunk = chunk.context("Stream read error")?;
        buf.push_str(&String::from_utf8_lossy(&chunk));
        drain_stream_buffer(&mut buf, &mut stats, start, &mut ttft, &mut stdout);
    }

    println!();

    let total = start.elapsed();
    let ttft = ttft.unwrap_or(total);
    print_timing_stats(
        model,
        ttft,
        total,
        stats.prompt_tokens,
        stats.completion_tokens,
    );

    Ok(())
}

fn drain_stream_buffer<W: Write>(
    buf: &mut String,
    stats: &mut StreamStats,
    start: Instant,
    ttft: &mut Option<Duration>,
    output: &mut W,
) {
    loop {
        let Some(newline_pos) = buf.find('\n') else {
            break;
        };
        let line = buf[..newline_pos].trim().to_owned();
        buf.drain(..=newline_pos);

        if let Some(content) = parse_stream_line(&line, stats) {
            if ttft.is_none() {
                *ttft = Some(start.elapsed());
            }
            write!(output, "{}", content).ok();
            output.flush().ok();
        }
    }
}

fn parse_stream_line(line: &str, stats: &mut StreamStats) -> Option<String> {
    if line.is_empty() || !line.starts_with("data: ") {
        return None;
    }

    let data = &line["data: ".len()..];
    if data == "[DONE]" {
        return None;
    }

    let Ok(value): Result<Value, _> = serde_json::from_str(data) else {
        return None;
    };

    update_stream_stats(&value, stats);
    stream_content(&value).map(str::to_string)
}

fn update_stream_stats(value: &Value, stats: &mut StreamStats) {
    let Some(usage) = value.get("usage") else {
        return;
    };

    stats.prompt_tokens = usage
        .get("prompt_tokens")
        .and_then(|value| value.as_u64())
        .unwrap_or(stats.prompt_tokens);
    stats.completion_tokens = usage
        .get("completion_tokens")
        .and_then(|value| value.as_u64())
        .unwrap_or(stats.completion_tokens);
}

fn stream_content(value: &Value) -> Option<&str> {
    let content = value
        .get("choices")?
        .get(0)?
        .get("delta")?
        .get("content")?
        .as_str()?;

    (!content.is_empty()).then_some(content)
}

#[cfg_attr(coverage_nightly, coverage(off))]
async fn query_blocking(
    client: reqwest::Client,
    url: String,
    api_key: String,
    body: ChatRequest,
    start: Instant,
    model: &str,
) -> Result<()> {
    let response = client
        .post(&url)
        .bearer_auth(&api_key)
        .json(&body)
        .send()
        .await
        .context("Failed to send request")?;

    if !response.status().is_success() {
        let status = response.status();
        let text = response.text().await.unwrap_or_default();
        bail!("API error {}: {}", status, text);
    }

    let resp: ChatResponse = response.json().await.context("Failed to parse response")?;
    let total = start.elapsed();

    if let Some(choice) = resp.choices.first() {
        if let Some(content) = &choice.message.content {
            println!("{}", content);
        }
    }

    let (prompt_tokens, completion_tokens) = resp
        .usage
        .map(|u| (u.prompt_tokens, u.completion_tokens))
        .unwrap_or((0, 0));

    print_timing_stats(model, total, total, prompt_tokens, completion_tokens);

    Ok(())
}

#[cfg_attr(coverage_nightly, coverage(off))]
fn print_timing_stats(
    model: &str,
    ttft: std::time::Duration,
    total: std::time::Duration,
    prompt_tokens: u64,
    completion_tokens: u64,
) {
    let total_secs = total.as_secs_f64();
    let ttft_secs = ttft.as_secs_f64();
    let gen_secs = (total_secs - ttft_secs).max(0.001);
    let speed = completion_tokens as f64 / gen_secs;

    let mut err = stderr();
    writeln!(err, "---").ok();
    writeln!(
        err,
        "ttft: {:.0}ms | total: {:.1}s | tokens: {} | speed: {:.1} tok/s",
        ttft.as_millis(),
        total_secs,
        completion_tokens,
        speed
    )
    .ok();
    writeln!(
        err,
        "model: {} | in: {} tok | out: {} tok",
        model, prompt_tokens, completion_tokens
    )
    .ok();
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    fn model(id: &str, name: &str) -> Model {
        Model {
            id: id.to_string(),
            name: name.to_string(),
            context_length: None,
            pricing: None,
        }
    }

    #[test]
    fn build_models_url_includes_optional_category() {
        assert_eq!(
            build_models_url(None),
            "https://openrouter.ai/api/v1/models"
        );
        assert_eq!(
            build_models_url(Some("programming")),
            "https://openrouter.ai/api/v1/models?category=programming"
        );
    }

    #[test]
    fn filter_models_matches_id_or_name_case_insensitively() {
        let models = vec![
            model("anthropic/claude-opus-4.5", "Claude Opus"),
            model("openai/gpt-5.5", "GPT"),
            model("deepseek/deepseek-v4-flash", "DeepSeek Flash"),
        ];

        let by_name = filter_models(models, Some("flash"));

        assert_eq!(by_name.len(), 1);
        assert_eq!(by_name[0].id, "deepseek/deepseek-v4-flash");
    }

    #[test]
    fn filter_models_without_search_returns_all_models() {
        let models = vec![model("openai/gpt-5.5", "GPT")];

        let filtered = filter_models(models, None);

        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].name, "GPT");
    }

    #[test]
    fn format_price_converts_token_price_to_million_tokens() {
        assert_eq!(format_price("0.000003"), "$3.00");
        assert_eq!(format_price("not-a-number"), "$0.00");
    }

    #[test]
    fn build_chat_request_adds_system_message_before_user_message() {
        let request = build_chat_request(
            "openai/gpt-5.5".to_string(),
            "Hello".to_string(),
            Some("Be terse".to_string()),
            128,
            true,
        );

        assert_eq!(request.model, "openai/gpt-5.5");
        assert_eq!(request.max_tokens, 128);
        assert!(request.stream);
        assert_eq!(
            request.messages,
            vec![
                Message {
                    role: "system".to_string(),
                    content: Some("Be terse".to_string()),
                },
                Message {
                    role: "user".to_string(),
                    content: Some("Hello".to_string()),
                },
            ]
        );
    }

    #[test]
    fn build_chat_request_omits_empty_system_message() {
        let request =
            build_chat_request("model".to_string(), "Prompt".to_string(), None, 64, false);

        assert_eq!(
            request.messages,
            vec![Message {
                role: "user".to_string(),
                content: Some("Prompt".to_string()),
            }]
        );
        assert!(!request.stream);
    }

    #[test]
    fn parse_stream_line_extracts_content_and_usage() {
        let mut stats = StreamStats::default();
        let line = r#"data: {"usage":{"prompt_tokens":5,"completion_tokens":8},"choices":[{"delta":{"content":"Hi"}}]}"#;

        let content = parse_stream_line(line, &mut stats);

        assert_eq!(content.as_deref(), Some("Hi"));
        assert_eq!(
            stats,
            StreamStats {
                prompt_tokens: 5,
                completion_tokens: 8,
            }
        );
    }

    #[test]
    fn parse_stream_line_ignores_non_data_and_done_lines() {
        let mut stats = StreamStats {
            prompt_tokens: 3,
            completion_tokens: 4,
        };

        assert_eq!(parse_stream_line("", &mut stats), None);
        assert_eq!(parse_stream_line("event: ping", &mut stats), None);
        assert_eq!(parse_stream_line("data: [DONE]", &mut stats), None);
        assert_eq!(
            stats,
            StreamStats {
                prompt_tokens: 3,
                completion_tokens: 4,
            }
        );
    }

    #[test]
    fn parse_stream_line_keeps_previous_usage_when_missing() {
        let mut stats = StreamStats {
            prompt_tokens: 3,
            completion_tokens: 4,
        };
        let line = r#"data: {"usage":{"prompt_tokens":9},"choices":[{"delta":{}}]}"#;

        let content = parse_stream_line(line, &mut stats);

        assert_eq!(content, None);
        assert_eq!(
            stats,
            StreamStats {
                prompt_tokens: 9,
                completion_tokens: 4,
            }
        );
    }

    #[test]
    fn drain_stream_buffer_preserves_partial_lines_and_writes_content() {
        let mut stats = StreamStats::default();
        let mut ttft = None;
        let mut output = Vec::new();
        let mut buf = concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"Hel\"}}]}\n",
            "data: {\"usage\":{\"prompt_tokens\":2,\"completion_tokens\":3},",
        )
        .to_string();

        drain_stream_buffer(&mut buf, &mut stats, Instant::now(), &mut ttft, &mut output);

        assert_eq!(String::from_utf8(output).unwrap(), "Hel");
        assert!(ttft.is_some());
        assert_eq!(stats, StreamStats::default());
        assert_eq!(
            buf,
            "data: {\"usage\":{\"prompt_tokens\":2,\"completion_tokens\":3},"
        );
    }

    #[test]
    fn cli_definition_is_valid() {
        Cli::command().debug_assert();
    }
}
