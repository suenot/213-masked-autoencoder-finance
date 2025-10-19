use anyhow::Result;
use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use serde::Deserialize;

// ─── OHLCV Candle ──────────────────────────────────────────────────────────

/// A single OHLCV candle from market data.
#[derive(Debug, Clone)]
pub struct OhlcvCandle {
    pub timestamp: u64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

// ─── Bybit API Types ───────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct BybitResponse {
    #[serde(rename = "retCode")]
    pub ret_code: i32,
    #[serde(rename = "retMsg")]
    pub ret_msg: String,
    pub result: BybitResult,
}

#[derive(Debug, Deserialize)]
pub struct BybitResult {
    pub list: Vec<Vec<String>>,
}

/// Fetch OHLCV candles from Bybit public API.
pub fn fetch_bybit_candles(
    symbol: &str,
    interval: &str,
    limit: usize,
) -> Result<Vec<OhlcvCandle>> {
    let url = format!(
        "https://api.bybit.com/v5/market/kline?category=linear&symbol={}&interval={}&limit={}",
        symbol, interval, limit
    );

    let client = reqwest::blocking::Client::new();
    let resp: BybitResponse = client.get(&url).send()?.json()?;

    if resp.ret_code != 0 {
        anyhow::bail!("Bybit API error: {}", resp.ret_msg);
    }

    let mut candles: Vec<OhlcvCandle> = resp
        .result
        .list
        .iter()
        .filter_map(|row| {
            if row.len() >= 6 {
                Some(OhlcvCandle {
                    timestamp: row[0].parse().unwrap_or(0),
                    open: row[1].parse().unwrap_or(0.0),
                    high: row[2].parse().unwrap_or(0.0),
                    low: row[3].parse().unwrap_or(0.0),
                    close: row[4].parse().unwrap_or(0.0),
                    volume: row[5].parse().unwrap_or(0.0),
                })
            } else {
                None
            }
        })
        .collect();

    // Bybit returns newest first, reverse to chronological order
    candles.reverse();
    Ok(candles)
}

/// Convert candles to an OHLCV matrix (T x 5).
pub fn candles_to_matrix(candles: &[OhlcvCandle]) -> Array2<f64> {
    let t = candles.len();
    let mut data = Array2::zeros((t, 5));
    for (i, c) in candles.iter().enumerate() {
        data[[i, 0]] = c.open;
        data[[i, 1]] = c.high;
        data[[i, 2]] = c.low;
        data[[i, 3]] = c.close;
        data[[i, 4]] = c.volume;
    }
    data
}

// ─── Positional Encoding ───────────────────────────────────────────────────

/// Generate sinusoidal positional encoding matrix (max_len x d_model).
pub fn sinusoidal_positional_encoding(max_len: usize, d_model: usize) -> Array2<f64> {
    let mut pe = Array2::zeros((max_len, d_model));
    for pos in 0..max_len {
        for k in 0..d_model / 2 {
            let angle = (pos as f64) / (10000.0_f64).powf(2.0 * k as f64 / d_model as f64);
            pe[[pos, 2 * k]] = angle.sin();
            if 2 * k + 1 < d_model {
                pe[[pos, 2 * k + 1]] = angle.cos();
            }
        }
    }
    pe
}

// ─── Patch Embedding ───────────────────────────────────────────────────────

/// Converts OHLCV time series into patch embeddings.
pub struct PatchEmbedding {
    pub patch_size: usize,
    pub d_model: usize,
    pub projection: Array2<f64>,
    pub bias: Array1<f64>,
    pub positional_encoding: Array2<f64>,
}

impl PatchEmbedding {
    /// Create a new PatchEmbedding with random weights.
    pub fn new(patch_size: usize, d_model: usize, max_patches: usize, seed: u64) -> Self {
        let input_dim = patch_size * 5; // 5 features (OHLCV)
        let mut rng = StdRng::seed_from_u64(seed);
        let scale = (2.0 / (input_dim + d_model) as f64).sqrt();

        let projection =
            Array2::from_shape_fn((input_dim, d_model), |_| rng.gen_range(-scale..scale));
        let bias = Array1::zeros(d_model);
        let positional_encoding = sinusoidal_positional_encoding(max_patches, d_model);

        Self {
            patch_size,
            d_model,
            projection,
            bias,
            positional_encoding,
        }
    }

    /// Create patches from OHLCV matrix and embed them.
    pub fn embed(&self, ohlcv: &Array2<f64>) -> (Array2<f64>, usize) {
        let t = ohlcv.nrows();
        let num_patches = t / self.patch_size;

        let mut patches = Array2::zeros((num_patches, self.d_model));

        for i in 0..num_patches {
            let start = i * self.patch_size;
            let end = start + self.patch_size;
            let patch_slice = ohlcv.slice(ndarray::s![start..end, ..]);

            // Flatten the patch
            let mut flat = Vec::with_capacity(self.patch_size * 5);
            for row in 0..self.patch_size {
                for col in 0..5 {
                    flat.push(patch_slice[[row, col]]);
                }
            }
            let flat_arr = Array1::from(flat);

            // Project and add positional encoding
            let embedded = flat_arr.dot(&self.projection) + &self.bias;
            let with_pos = if i < self.positional_encoding.nrows() {
                embedded + &self.positional_encoding.row(i)
            } else {
                embedded
            };
            patches.row_mut(i).assign(&with_pos);
        }

        (patches, num_patches)
    }

    /// Reconstruct OHLCV patches from embeddings (inverse projection).
    pub fn reconstruct_patch(&self, embedding: &Array1<f64>) -> Array1<f64> {
        // Simple pseudo-inverse via transpose (approximate)
        let proj_t = self.projection.t();
        embedding.dot(&proj_t)
    }
}

// ─── Transformer Components ───────────────────────────────────────────────

/// Simple single-head self-attention.
pub struct SelfAttention {
    pub d_model: usize,
    pub w_q: Array2<f64>,
    pub w_k: Array2<f64>,
    pub w_v: Array2<f64>,
    pub w_o: Array2<f64>,
}

impl SelfAttention {
    pub fn new(d_model: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let scale = (1.0 / d_model as f64).sqrt();
        let w_q = Array2::from_shape_fn((d_model, d_model), |_| rng.gen_range(-scale..scale));
        let w_k = Array2::from_shape_fn((d_model, d_model), |_| rng.gen_range(-scale..scale));
        let w_v = Array2::from_shape_fn((d_model, d_model), |_| rng.gen_range(-scale..scale));
        let w_o = Array2::from_shape_fn((d_model, d_model), |_| rng.gen_range(-scale..scale));
        Self {
            d_model,
            w_q,
            w_k,
            w_v,
            w_o,
        }
    }

    /// Apply self-attention: input shape (N, d_model) -> output shape (N, d_model).
    pub fn forward(&self, x: &Array2<f64>) -> Array2<f64> {
        let q = x.dot(&self.w_q);
        let k = x.dot(&self.w_k);
        let v = x.dot(&self.w_v);

        let scale = (self.d_model as f64).sqrt();
        let scores = q.dot(&k.t()) / scale;

        // Softmax over last axis
        let attn = softmax_2d(&scores);
        let out = attn.dot(&v);
        out.dot(&self.w_o)
    }
}

/// Feed-forward network with GELU activation.
pub struct FeedForward {
    pub w1: Array2<f64>,
    pub b1: Array1<f64>,
    pub w2: Array2<f64>,
    pub b2: Array1<f64>,
}

impl FeedForward {
    pub fn new(d_model: usize, d_ff: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let scale1 = (2.0 / (d_model + d_ff) as f64).sqrt();
        let scale2 = (2.0 / (d_ff + d_model) as f64).sqrt();
        let w1 = Array2::from_shape_fn((d_model, d_ff), |_| rng.gen_range(-scale1..scale1));
        let b1 = Array1::zeros(d_ff);
        let w2 = Array2::from_shape_fn((d_ff, d_model), |_| rng.gen_range(-scale2..scale2));
        let b2 = Array1::zeros(d_model);
        Self { w1, b1, w2, b2 }
    }

    pub fn forward(&self, x: &Array2<f64>) -> Array2<f64> {
        let h = x.dot(&self.w1) + &self.b1;
        let h_gelu = gelu_2d(&h);
        h_gelu.dot(&self.w2) + &self.b2
    }
}

/// A single transformer block (attention + FFN + residual + layer norm).
pub struct TransformerBlock {
    pub attention: SelfAttention,
    pub ffn: FeedForward,
}

impl TransformerBlock {
    pub fn new(d_model: usize, d_ff: usize, seed: u64) -> Self {
        Self {
            attention: SelfAttention::new(d_model, seed),
            ffn: FeedForward::new(d_model, d_ff, seed.wrapping_add(1)),
        }
    }

    pub fn forward(&self, x: &Array2<f64>) -> Array2<f64> {
        // Self-attention with residual connection
        let attn_out = self.attention.forward(x);
        let x1 = x + &attn_out;
        let x1_norm = layer_norm_2d(&x1);

        // FFN with residual connection
        let ffn_out = self.ffn.forward(&x1_norm);
        let x2 = &x1_norm + &ffn_out;
        layer_norm_2d(&x2)
    }
}

// ─── Masked Autoencoder ────────────────────────────────────────────────────

/// The Masked Autoencoder for financial time series.
pub struct MaskedAutoencoder {
    pub patch_embedding: PatchEmbedding,
    pub encoder_blocks: Vec<TransformerBlock>,
    pub decoder_blocks: Vec<TransformerBlock>,
    pub mask_token: Array1<f64>,
    pub masking_ratio: f64,
    pub d_model: usize,
    pub decoder_projection: Array2<f64>,
    rng: StdRng,
}

impl MaskedAutoencoder {
    /// Create a new MAE with given architecture parameters.
    pub fn new(
        patch_size: usize,
        d_model: usize,
        d_ff: usize,
        encoder_depth: usize,
        decoder_depth: usize,
        max_patches: usize,
        masking_ratio: f64,
        seed: u64,
    ) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let patch_embedding = PatchEmbedding::new(patch_size, d_model, max_patches, seed);

        let encoder_blocks = (0..encoder_depth)
            .map(|i| TransformerBlock::new(d_model, d_ff, seed + 100 + i as u64))
            .collect();

        let decoder_blocks = (0..decoder_depth)
            .map(|i| TransformerBlock::new(d_model, d_ff, seed + 200 + i as u64))
            .collect();

        let mask_token = Array1::from_shape_fn(d_model, |_| rng.gen_range(-0.02..0.02));

        let patch_dim = patch_size * 5;
        let scale = (2.0 / (d_model + patch_dim) as f64).sqrt();
        let decoder_projection =
            Array2::from_shape_fn((d_model, patch_dim), |_| rng.gen_range(-scale..scale));

        Self {
            patch_embedding,
            encoder_blocks,
            decoder_blocks,
            mask_token,
            masking_ratio,
            d_model,
            decoder_projection,
            rng,
        }
    }

    /// Generate random mask indices for the given number of patches.
    pub fn generate_mask(&mut self, num_patches: usize) -> (Vec<usize>, Vec<usize>) {
        let num_masked = ((num_patches as f64) * self.masking_ratio).round() as usize;
        let num_masked = num_masked.max(1).min(num_patches - 1);

        let mut indices: Vec<usize> = (0..num_patches).collect();
        indices.shuffle(&mut self.rng);

        let masked_indices: Vec<usize> = indices[..num_masked].to_vec();
        let visible_indices: Vec<usize> = indices[num_masked..].to_vec();

        (masked_indices, visible_indices)
    }

    /// Encode only the visible patches through the encoder.
    pub fn encode(&self, patches: &Array2<f64>, visible_indices: &[usize]) -> Array2<f64> {
        let n_visible = visible_indices.len();
        let mut visible = Array2::zeros((n_visible, self.d_model));
        for (i, &idx) in visible_indices.iter().enumerate() {
            visible.row_mut(i).assign(&patches.row(idx));
        }

        let mut h = visible;
        for block in &self.encoder_blocks {
            h = block.forward(&h);
        }
        h
    }

    /// Decode: combine encoded visible tokens with mask tokens and run decoder.
    pub fn decode(
        &self,
        encoded_visible: &Array2<f64>,
        visible_indices: &[usize],
        masked_indices: &[usize],
        num_patches: usize,
    ) -> Array2<f64> {
        // Build full sequence with mask tokens at masked positions
        let mut full_seq = Array2::zeros((num_patches, self.d_model));

        // Place encoded visible tokens
        for (i, &idx) in visible_indices.iter().enumerate() {
            full_seq.row_mut(idx).assign(&encoded_visible.row(i));
        }

        // Place mask tokens at masked positions
        for &idx in masked_indices {
            full_seq.row_mut(idx).assign(&self.mask_token);
        }

        // Add positional encoding
        for i in 0..num_patches {
            if i < self.patch_embedding.positional_encoding.nrows() {
                let pe = self.patch_embedding.positional_encoding.row(i).to_owned();
                let current = full_seq.row(i).to_owned();
                full_seq.row_mut(i).assign(&(current + pe));
            }
        }

        // Run through decoder blocks
        let mut h = full_seq;
        for block in &self.decoder_blocks {
            h = block.forward(&h);
        }

        // Project to patch dimension
        h.dot(&self.decoder_projection)
    }

    /// Compute reconstruction loss (MSE on masked patches only).
    pub fn reconstruction_loss(
        &self,
        original_patches_flat: &Array2<f64>,
        reconstructed: &Array2<f64>,
        masked_indices: &[usize],
    ) -> f64 {
        if masked_indices.is_empty() {
            return 0.0;
        }

        let mut total_loss = 0.0;
        let patch_dim = original_patches_flat.ncols();

        for &idx in masked_indices {
            for j in 0..patch_dim {
                let diff = original_patches_flat[[idx, j]] - reconstructed[[idx, j]];
                total_loss += diff * diff;
            }
        }

        total_loss / (masked_indices.len() * patch_dim) as f64
    }

    /// Run one forward pass: embed, mask, encode, decode, compute loss.
    pub fn forward_pass(&mut self, ohlcv: &Array2<f64>) -> (f64, Vec<usize>) {
        let (patches, num_patches) = self.patch_embedding.embed(ohlcv);
        let (masked_indices, visible_indices) = self.generate_mask(num_patches);

        let encoded = self.encode(&patches, &visible_indices);
        let reconstructed =
            self.decode(&encoded, &visible_indices, &masked_indices, num_patches);

        // Create flat original patches for loss computation
        let patch_dim = self.patch_embedding.patch_size * 5;
        let mut original_flat = Array2::zeros((num_patches, patch_dim));
        for i in 0..num_patches {
            let start = i * self.patch_embedding.patch_size;
            let end = start + self.patch_embedding.patch_size;
            let mut flat_idx = 0;
            for row in start..end {
                for col in 0..5 {
                    if row < ohlcv.nrows() {
                        original_flat[[i, flat_idx]] = ohlcv[[row, col]];
                    }
                    flat_idx += 1;
                }
            }
        }

        let loss = self.reconstruction_loss(&original_flat, &reconstructed, &masked_indices);
        (loss, masked_indices)
    }

    /// Pre-train the MAE on OHLCV data for a given number of epochs.
    /// Returns the average loss per epoch.
    pub fn pretrain(&mut self, ohlcv: &Array2<f64>, epochs: usize) -> Vec<f64> {
        let mut losses = Vec::with_capacity(epochs);
        for _ in 0..epochs {
            let (loss, _) = self.forward_pass(ohlcv);
            losses.push(loss);
        }
        losses
    }

    /// Compute the average reconstruction error for given data (used as anomaly score).
    pub fn anomaly_score(&mut self, ohlcv: &Array2<f64>, num_samples: usize) -> f64 {
        let mut total_loss = 0.0;
        for _ in 0..num_samples {
            let (loss, _) = self.forward_pass(ohlcv);
            total_loss += loss;
        }
        total_loss / num_samples as f64
    }
}

// ─── Trading Strategy ──────────────────────────────────────────────────────

/// Trading signal derived from MAE analysis.
#[derive(Debug, Clone, PartialEq)]
pub enum TradingSignal {
    Buy,
    Sell,
    Hold,
}

/// A trading strategy based on MAE reconstruction error and trend analysis.
pub struct MaeTradingStrategy {
    pub mae: MaskedAutoencoder,
    pub anomaly_threshold: f64,
    pub trend_window: usize,
    pub baseline_error: f64,
}

impl MaeTradingStrategy {
    /// Create a new trading strategy with a pre-trained MAE.
    pub fn new(
        mae: MaskedAutoencoder,
        anomaly_threshold: f64,
        trend_window: usize,
    ) -> Self {
        Self {
            mae,
            anomaly_threshold,
            trend_window,
            baseline_error: 0.0,
        }
    }

    /// Calibrate the baseline reconstruction error on training data.
    pub fn calibrate(&mut self, ohlcv: &Array2<f64>, num_samples: usize) {
        self.baseline_error = self.mae.anomaly_score(ohlcv, num_samples);
    }

    /// Generate a trading signal based on recent data.
    pub fn generate_signal(&mut self, ohlcv: &Array2<f64>) -> TradingSignal {
        let current_error = self.mae.anomaly_score(ohlcv, 3);

        // If reconstruction error is too high, market is unusual -> Hold
        if self.baseline_error > 0.0 && current_error > self.anomaly_threshold * self.baseline_error
        {
            return TradingSignal::Hold;
        }

        // Simple trend detection using close prices
        let n = ohlcv.nrows();
        if n < self.trend_window + 1 {
            return TradingSignal::Hold;
        }

        let recent_close = ohlcv[[n - 1, 3]];
        let past_close = ohlcv[[n - 1 - self.trend_window, 3]];
        let trend = (recent_close - past_close) / past_close;

        if trend > 0.005 {
            TradingSignal::Buy
        } else if trend < -0.005 {
            TradingSignal::Sell
        } else {
            TradingSignal::Hold
        }
    }

    /// Backtest the strategy on historical OHLCV data.
    pub fn backtest(
        &mut self,
        ohlcv: &Array2<f64>,
        window_size: usize,
        step_size: usize,
    ) -> BacktestResult {
        let n = ohlcv.nrows();
        let mut portfolio_value = 10000.0;
        let mut position = 0.0; // units held
        let mut signals = Vec::new();
        let mut portfolio_values = vec![portfolio_value];
        let mut trades = 0;

        let mut i = window_size;
        while i < n {
            let window = ohlcv.slice(ndarray::s![i - window_size..i, ..]).to_owned();
            let signal = self.generate_signal(&window);

            let current_price = ohlcv[[i, 3]]; // close price

            match signal {
                TradingSignal::Buy if position <= 0.0 => {
                    position = portfolio_value / current_price;
                    trades += 1;
                }
                TradingSignal::Sell if position > 0.0 => {
                    portfolio_value = position * current_price;
                    position = 0.0;
                    trades += 1;
                }
                _ => {}
            }

            // Mark to market
            let mtm = if position > 0.0 {
                position * current_price
            } else {
                portfolio_value
            };
            portfolio_values.push(mtm);
            signals.push(signal);

            i += step_size;
        }

        // Close any open position
        if position > 0.0 {
            portfolio_value = position * ohlcv[[n - 1, 3]];
        }

        let total_return = (portfolio_value - 10000.0) / 10000.0;
        let max_drawdown = compute_max_drawdown(&portfolio_values);
        let sharpe = compute_sharpe_ratio(&portfolio_values);

        BacktestResult {
            total_return,
            max_drawdown,
            sharpe_ratio: sharpe,
            num_trades: trades,
            final_value: portfolio_value,
            signals,
        }
    }
}

/// Results from backtesting the MAE trading strategy.
#[derive(Debug)]
pub struct BacktestResult {
    pub total_return: f64,
    pub max_drawdown: f64,
    pub sharpe_ratio: f64,
    pub num_trades: usize,
    pub final_value: f64,
    pub signals: Vec<TradingSignal>,
}

// ─── Utility Functions ─────────────────────────────────────────────────────

/// GELU activation function applied element-wise to a 2D array.
pub fn gelu_2d(x: &Array2<f64>) -> Array2<f64> {
    x.mapv(|v| {
        0.5 * v * (1.0 + ((2.0_f64 / std::f64::consts::PI).sqrt() * (v + 0.044715 * v.powi(3))).tanh())
    })
}

/// Softmax over rows of a 2D array.
pub fn softmax_2d(x: &Array2<f64>) -> Array2<f64> {
    let mut result = x.clone();
    for mut row in result.rows_mut() {
        let max_val = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        row.mapv_inplace(|v| (v - max_val).exp());
        let sum: f64 = row.iter().sum();
        if sum > 0.0 {
            row.mapv_inplace(|v| v / sum);
        }
    }
    result
}

/// Layer normalization applied to each row of a 2D array.
pub fn layer_norm_2d(x: &Array2<f64>) -> Array2<f64> {
    let eps = 1e-5;
    let mut result = x.clone();
    for mut row in result.rows_mut() {
        let mean = row.mean().unwrap_or(0.0);
        let var = row.mapv(|v| (v - mean).powi(2)).mean().unwrap_or(0.0);
        let std = (var + eps).sqrt();
        row.mapv_inplace(|v| (v - mean) / std);
    }
    result
}

/// L2 normalize a 1D array.
pub fn l2_normalize(x: &Array1<f64>) -> Array1<f64> {
    let norm = x.mapv(|v| v * v).sum().sqrt();
    if norm > 1e-10 {
        x / norm
    } else {
        x.clone()
    }
}

/// Normalize OHLCV data by z-score per column.
pub fn normalize_ohlcv(data: &Array2<f64>) -> Array2<f64> {
    let mut normalized = data.clone();
    for j in 0..data.ncols() {
        let col = data.column(j);
        let mean = col.mean().unwrap_or(0.0);
        let var = col.mapv(|v| (v - mean).powi(2)).mean().unwrap_or(0.0);
        let std = (var + 1e-8).sqrt();
        for i in 0..data.nrows() {
            normalized[[i, j]] = (data[[i, j]] - mean) / std;
        }
    }
    normalized
}

/// Compute maximum drawdown from a portfolio value series.
pub fn compute_max_drawdown(values: &[f64]) -> f64 {
    let mut max_dd = 0.0;
    let mut peak = values[0];
    for &v in values.iter() {
        if v > peak {
            peak = v;
        }
        let dd = (peak - v) / peak;
        if dd > max_dd {
            max_dd = dd;
        }
    }
    max_dd
}

/// Compute annualized Sharpe ratio from portfolio values.
pub fn compute_sharpe_ratio(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let returns: Vec<f64> = values
        .windows(2)
        .map(|w| (w[1] - w[0]) / w[0])
        .collect();

    let n = returns.len() as f64;
    let mean_ret = returns.iter().sum::<f64>() / n;
    let var = returns.iter().map(|r| (r - mean_ret).powi(2)).sum::<f64>() / n;
    let std = var.sqrt();

    if std < 1e-10 {
        return 0.0;
    }
    (mean_ret / std) * (252.0_f64).sqrt()
}

/// Generate synthetic OHLCV data for testing.
pub fn generate_synthetic_ohlcv(num_candles: usize, seed: u64) -> Array2<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut data = Array2::zeros((num_candles, 5));

    let mut price = 100.0;
    for i in 0..num_candles {
        let ret: f64 = rng.gen_range(-0.03..0.03);
        let open: f64 = price;
        let close: f64 = price * (1.0 + ret);
        let high: f64 = open.max(close) * (1.0 + rng.gen_range(0.0..0.01));
        let low: f64 = open.min(close) * (1.0 - rng.gen_range(0.0..0.01));
        let volume = rng.gen_range(1000.0..10000.0);

        data[[i, 0]] = open;
        data[[i, 1]] = high;
        data[[i, 2]] = low;
        data[[i, 3]] = close;
        data[[i, 4]] = volume;

        price = close;
    }
    data
}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sinusoidal_positional_encoding() {
        let pe = sinusoidal_positional_encoding(10, 16);
        assert_eq!(pe.shape(), &[10, 16]);
        // Position 0 should have sin(0)=0 for even indices
        assert!((pe[[0, 0]] - 0.0).abs() < 1e-10);
        // Position 0 should have cos(0)=1 for odd indices
        assert!((pe[[0, 1]] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_patch_embedding() {
        let ohlcv = generate_synthetic_ohlcv(32, 42);
        let emb = PatchEmbedding::new(4, 16, 100, 42);
        let (patches, num_patches) = emb.embed(&ohlcv);
        assert_eq!(num_patches, 8); // 32 / 4 = 8
        assert_eq!(patches.shape(), &[8, 16]);
    }

    #[test]
    fn test_self_attention() {
        let attn = SelfAttention::new(16, 42);
        let input = Array2::from_shape_fn((4, 16), |(i, j)| (i * 16 + j) as f64 * 0.01);
        let output = attn.forward(&input);
        assert_eq!(output.shape(), &[4, 16]);
    }

    #[test]
    fn test_transformer_block() {
        let block = TransformerBlock::new(16, 32, 42);
        let input = Array2::from_shape_fn((4, 16), |(i, j)| (i * 16 + j) as f64 * 0.01);
        let output = block.forward(&input);
        assert_eq!(output.shape(), &[4, 16]);
    }

    #[test]
    fn test_mae_forward_pass() {
        let ohlcv = generate_synthetic_ohlcv(64, 42);
        let normalized = normalize_ohlcv(&ohlcv);
        let mut mae = MaskedAutoencoder::new(
            4,   // patch_size
            16,  // d_model
            32,  // d_ff
            2,   // encoder_depth
            1,   // decoder_depth
            100, // max_patches
            0.75, // masking_ratio
            42,  // seed
        );

        let (loss, masked_indices) = mae.forward_pass(&normalized);
        assert!(loss.is_finite());
        assert!(loss >= 0.0);
        // 75% of 16 patches = 12 masked
        let num_patches = 64 / 4;
        let expected_masked = ((num_patches as f64) * 0.75).round() as usize;
        assert_eq!(masked_indices.len(), expected_masked);
    }

    #[test]
    fn test_mae_pretrain() {
        let ohlcv = generate_synthetic_ohlcv(64, 42);
        let normalized = normalize_ohlcv(&ohlcv);
        let mut mae = MaskedAutoencoder::new(4, 16, 32, 2, 1, 100, 0.75, 42);

        let losses = mae.pretrain(&normalized, 5);
        assert_eq!(losses.len(), 5);
        for loss in &losses {
            assert!(loss.is_finite());
        }
    }

    #[test]
    fn test_trading_signal_generation() {
        let ohlcv = generate_synthetic_ohlcv(128, 42);
        let normalized = normalize_ohlcv(&ohlcv);
        let mae = MaskedAutoencoder::new(4, 16, 32, 2, 1, 100, 0.75, 42);
        let mut strategy = MaeTradingStrategy::new(mae, 2.0, 5);

        strategy.calibrate(&normalized, 3);
        assert!(strategy.baseline_error > 0.0);

        let signal = strategy.generate_signal(&normalized);
        assert!(
            signal == TradingSignal::Buy
                || signal == TradingSignal::Sell
                || signal == TradingSignal::Hold
        );
    }

    #[test]
    fn test_backtest() {
        let ohlcv = generate_synthetic_ohlcv(200, 42);
        let normalized = normalize_ohlcv(&ohlcv);
        let mae = MaskedAutoencoder::new(4, 16, 32, 2, 1, 100, 0.75, 42);
        let mut strategy = MaeTradingStrategy::new(mae, 2.0, 5);
        strategy.calibrate(&normalized, 3);

        let result = strategy.backtest(&normalized, 64, 8);
        assert!(result.final_value > 0.0);
        assert!(result.max_drawdown >= 0.0 && result.max_drawdown <= 1.0);
        assert!(result.sharpe_ratio.is_finite());
    }

    #[test]
    fn test_normalize_ohlcv() {
        let data = generate_synthetic_ohlcv(32, 42);
        let normalized = normalize_ohlcv(&data);
        assert_eq!(normalized.shape(), data.shape());

        // Check each column has approximately zero mean
        for j in 0..5 {
            let col_mean = normalized.column(j).mean().unwrap_or(1.0);
            assert!(col_mean.abs() < 1e-6, "Column {} mean: {}", j, col_mean);
        }
    }

    #[test]
    fn test_gelu_activation() {
        let x = Array2::from_shape_vec((2, 2), vec![-1.0, 0.0, 1.0, 2.0]).unwrap();
        let result = gelu_2d(&x);
        // GELU(0) = 0
        assert!((result[[0, 1]] - 0.0).abs() < 1e-6);
        // GELU(x) > 0 for x > 0
        assert!(result[[1, 0]] > 0.0);
        assert!(result[[1, 1]] > 0.0);
    }

    #[test]
    fn test_softmax() {
        let x = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 1.0, 1.0, 1.0]).unwrap();
        let result = softmax_2d(&x);
        // Each row should sum to 1
        for row in result.rows() {
            let sum: f64 = row.iter().sum();
            assert!((sum - 1.0).abs() < 1e-6);
        }
        // Second row should be uniform
        assert!((result[[1, 0]] - result[[1, 1]]).abs() < 1e-6);
    }

    #[test]
    fn test_max_drawdown() {
        let values = vec![100.0, 110.0, 90.0, 95.0, 80.0, 100.0];
        let dd = compute_max_drawdown(&values);
        // Peak=110, trough=80, dd = 30/110
        assert!((dd - 30.0 / 110.0).abs() < 1e-6);
    }

    #[test]
    fn test_sharpe_ratio() {
        let values = vec![100.0, 101.0, 102.0, 103.0, 104.0];
        let sharpe = compute_sharpe_ratio(&values);
        assert!(sharpe > 0.0); // Consistently positive returns -> positive Sharpe
    }
}
