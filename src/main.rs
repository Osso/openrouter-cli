use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use serde::Deserialize;

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
        /// Filter by category (programming, roleplay, marketing, etc.)
        #[arg(short, long)]
        category: Option<String>,

        /// Search/filter by name
        #[arg(short, long)]
        search: Option<String>,

        /// Show pricing info
        #[arg(short, long)]
        pricing: bool,
    },
}

#[derive(Debug, Deserialize)]
struct ModelsResponse {
    data: Vec<Model>,
}

#[derive(Debug, Deserialize)]
struct Model {
    id: String,
    #[allow(dead_code)]
    name: String,
    #[allow(dead_code)]
    description: Option<String>,
    context_length: Option<u64>,
    pricing: Option<Pricing>,
}

#[derive(Debug, Deserialize)]
struct Pricing {
    prompt: String,
    completion: String,
}

fn format_price(price_str: &str) -> String {
    let price: f64 = price_str.parse().unwrap_or(0.0);
    let per_million = price * 1_000_000.0;
    format!("${:.2}", per_million)
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Models {
            category,
            search,
            pricing,
        } => list_models(category, search, pricing).await?,
    }

    Ok(())
}

async fn list_models(category: Option<String>, search: Option<String>, show_pricing: bool) -> Result<()> {
    let client = reqwest::Client::new();
    let mut url = format!("{}/models", API_BASE);

    if let Some(cat) = &category {
        url = format!("{}?category={}", url, cat);
    }

    let resp: ModelsResponse = client
        .get(&url)
        .send()
        .await
        .context("Failed to fetch models")?
        .json()
        .await
        .context("Failed to parse response")?;

    let models: Vec<_> = resp
        .data
        .into_iter()
        .filter(|m| {
            search
                .as_ref()
                .map(|s| m.name.to_lowercase().contains(&s.to_lowercase()) || m.id.to_lowercase().contains(&s.to_lowercase()))
                .unwrap_or(true)
        })
        .collect();

    println!("Found {} models\n", models.len());

    for model in models {
        print!("{:50}", model.id);

        if let Some(ctx) = model.context_length {
            print!("  {:>7}k ctx", ctx / 1000);
        } else {
            print!("  {:>10}", "");
        }

        if show_pricing {
            if let Some(p) = &model.pricing {
                print!("  {:>8}/M in  {:>8}/M out", format_price(&p.prompt), format_price(&p.completion));
            }
        }

        println!();
    }

    Ok(())
}
