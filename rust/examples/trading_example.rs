use anyhow::Result;
use masked_autoencoder_finance::*;

fn main() -> Result<()> {
    println!("=== Masked Autoencoder for Financial Trading ===\n");

    // --- Step 1: Fetch market data from Bybit ---
    println!("[1] Fetching BTCUSDT candles from Bybit...");
    let candles = match fetch_bybit_candles("BTCUSDT", "60", 200) {
        Ok(c) => {
            println!("    Fetched {} candles", c.len());
            if let Some(first) = c.first() {
                println!(
                    "    First candle: O={:.2} H={:.2} L={:.2} C={:.2} V={:.2}",
                    first.open, first.high, first.low, first.close, first.volume
                );
            }
            candles_to_matrix(&c)
        }
        Err(e) => {
            println!("    Could not fetch from Bybit: {}", e);
            println!("    Using synthetic data instead.\n");
            generate_synthetic_ohlcv(200, 42)
        }
    };

    // --- Step 2: Normalize data ---
    println!("\n[2] Normalizing OHLCV data...");
    let normalized = normalize_ohlcv(&candles);
    println!("    Data shape: {} x {}", normalized.nrows(), normalized.ncols());

    // --- Step 3: Create and pre-train MAE ---
    println!("\n[3] Creating Masked Autoencoder...");
    let patch_size = 4;
    let d_model = 32;
    let d_ff = 64;
    let encoder_depth = 2;
    let decoder_depth = 1;
    let max_patches = 100;
    let masking_ratio = 0.75;
    let seed = 42;

    let mut mae = MaskedAutoencoder::new(
        patch_size,
        d_model,
        d_ff,
        encoder_depth,
        decoder_depth,
        max_patches,
        masking_ratio,
        seed,
    );

    println!("    Patch size: {}", patch_size);
    println!("    Model dimension: {}", d_model);
    println!("    Masking ratio: {}%", (masking_ratio * 100.0) as usize);
    println!("    Encoder depth: {} blocks", encoder_depth);
    println!("    Decoder depth: {} blocks", decoder_depth);

    println!("\n[4] Pre-training MAE (10 epochs)...");
    let losses = mae.pretrain(&normalized, 10);
    for (i, loss) in losses.iter().enumerate() {
        println!("    Epoch {}: loss = {:.6}", i + 1, loss);
    }

    // --- Step 4: Compute anomaly score ---
    println!("\n[5] Computing anomaly score...");
    let anomaly = mae.anomaly_score(&normalized, 5);
    println!("    Average reconstruction error: {:.6}", anomaly);

    // --- Step 5: Run trading strategy ---
    println!("\n[6] Running MAE Trading Strategy...");
    let mut strategy = MaeTradingStrategy::new(mae, 2.0, 5);

    println!("    Calibrating baseline error...");
    strategy.calibrate(&normalized, 5);
    println!("    Baseline error: {:.6}", strategy.baseline_error);

    println!("    Generating current signal...");
    let signal = strategy.generate_signal(&normalized);
    println!("    Current signal: {:?}", signal);

    // --- Step 6: Backtest ---
    println!("\n[7] Backtesting strategy...");
    let result = strategy.backtest(&normalized, 64, 4);
    println!("    Total return:  {:.2}%", result.total_return * 100.0);
    println!("    Max drawdown:  {:.2}%", result.max_drawdown * 100.0);
    println!("    Sharpe ratio:  {:.4}", result.sharpe_ratio);
    println!("    Num trades:    {}", result.num_trades);
    println!("    Final value:   ${:.2}", result.final_value);

    // Signal distribution
    let buys = result.signals.iter().filter(|s| **s == TradingSignal::Buy).count();
    let sells = result.signals.iter().filter(|s| **s == TradingSignal::Sell).count();
    let holds = result.signals.iter().filter(|s| **s == TradingSignal::Hold).count();
    println!("\n    Signal distribution:");
    println!("      Buy:  {}", buys);
    println!("      Sell: {}", sells);
    println!("      Hold: {}", holds);

    // --- Step 7: Fetch and analyze another pair ---
    println!("\n[8] Fetching ETHUSDT for cross-asset analysis...");
    match fetch_bybit_candles("ETHUSDT", "60", 200) {
        Ok(eth_candles) => {
            let eth_matrix = candles_to_matrix(&eth_candles);
            let eth_normalized = normalize_ohlcv(&eth_matrix);
            let eth_anomaly = strategy.mae.anomaly_score(&eth_normalized, 5);
            println!("    ETH anomaly score: {:.6}", eth_anomaly);
            println!(
                "    Ratio to BTC baseline: {:.2}x",
                eth_anomaly / strategy.baseline_error.max(1e-10)
            );
        }
        Err(e) => {
            println!("    Could not fetch ETHUSDT: {}", e);
        }
    }

    println!("\n=== Done ===");
    Ok(())
}
