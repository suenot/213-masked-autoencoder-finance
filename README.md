# Chapter 282: Masked Autoencoders for Financial Time Series

## Introduction

Masked Autoencoders (MAE) have emerged as a powerful self-supervised pre-training strategy, originally popularized in computer vision by He et al. (2021). The core idea is remarkably simple: randomly mask a large portion of input data, then train a model to reconstruct the missing pieces. This forces the model to learn rich, generalizable representations of the underlying data distribution without requiring any labels.

In finance, labeled data is notoriously scarce and expensive, while unlabeled price and volume data is abundant. MAE provides a natural bridge: pre-train on vast amounts of unlabeled market data, then fine-tune on small labeled datasets for downstream tasks such as trend prediction, volatility forecasting, or anomaly detection. This chapter explores the theoretical foundations of MAE, adapts the architecture for financial time series, and provides a complete Rust implementation with Bybit market data integration.

## Theoretical Foundations

### The Masked Autoencoder Framework

A Masked Autoencoder operates on a sequence of input tokens (or patches). Given an input sequence $\mathbf{X} = [x_1, x_2, \ldots, x_T]$, the MAE process involves three stages:

**1. Masking**: A random subset $\mathcal{M} \subset \{1, 2, \ldots, T\}$ of positions is selected for masking. The masking ratio $r = |\mathcal{M}| / T$ is typically high (60-80%). The visible set is $\mathcal{V} = \{1, \ldots, T\} \setminus \mathcal{M}$.

**2. Encoding**: Only the visible (unmasked) tokens are passed through the encoder:

$$\mathbf{H}_\mathcal{V} = f_{\text{enc}}(\mathbf{X}_\mathcal{V})$$

This asymmetric design (encoding only visible tokens) dramatically reduces computation during pre-training.

**3. Decoding**: The decoder takes the encoded visible tokens plus learnable mask tokens $\mathbf{m}$ and reconstructs the full sequence:

$$\hat{\mathbf{X}} = f_{\text{dec}}([\mathbf{H}_\mathcal{V}; \mathbf{m}_\mathcal{M}])$$

### Reconstruction Loss

The training objective minimizes the Mean Squared Error (MSE) over the masked positions only:

$$\mathcal{L}_{\text{MAE}} = \frac{1}{|\mathcal{M}|} \sum_{i \in \mathcal{M}} \| x_i - \hat{x}_i \|^2$$

Computing loss only on masked positions forces the model to truly predict missing information rather than simply copying visible inputs.

### Positional Encoding

Position information is critical for time series data. We use sinusoidal positional encoding:

$$PE_{(pos, 2k)} = \sin\left(\frac{pos}{10000^{2k/d}}\right)$$
$$PE_{(pos, 2k+1)} = \cos\left(\frac{pos}{10000^{2k/d}}\right)$$

where $pos$ is the position index and $k$ is the dimension index.

### Adaptation for Financial Time Series

Financial time series present unique challenges compared to images:

1. **Temporal ordering matters**: Unlike image patches where spatial relationships are relatively uniform, financial data has strict temporal causality.

2. **Multi-scale patterns**: Markets exhibit patterns at multiple time scales (intraday, daily, weekly, monthly).

3. **Non-stationarity**: Statistical properties of financial data change over time, requiring robust feature extraction.

4. **Feature heterogeneity**: OHLCV data combines price levels, ranges, and volume information with different scales and distributions.

Our financial MAE adaptation addresses these by:
- Using **overlapping temporal patches** to capture local structure
- **Normalizing features** within each patch to handle non-stationarity
- Applying **learnable linear projections** for patch embedding
- Using a **high masking ratio (75%)** to prevent trivial interpolation

### Patch Construction for OHLCV Data

Given OHLCV time series $\mathbf{X} \in \mathbb{R}^{T \times 5}$, we create patches of size $P$:

$$\text{patch}_i = \text{flatten}(\mathbf{X}[iP : (i+1)P, :]) \in \mathbb{R}^{5P}$$

Each patch is then projected to the model dimension $d$:

$$\mathbf{e}_i = \mathbf{W}_{\text{proj}} \cdot \text{patch}_i + \mathbf{b}_{\text{proj}} + \mathbf{PE}_i$$

where $\mathbf{W}_{\text{proj}} \in \mathbb{R}^{d \times 5P}$ is a learnable projection matrix.

## Architecture Details

### Encoder

The encoder consists of $L_e$ transformer blocks operating on visible tokens only:

$$\mathbf{H}^{(l+1)} = \text{TransformerBlock}(\mathbf{H}^{(l)})$$

Each transformer block contains:
- Multi-Head Self-Attention (MHSA)
- Feed-Forward Network (FFN) with GELU activation
- Layer normalization and residual connections

### Decoder

The decoder is a lightweight network with $L_d$ transformer blocks ($L_d < L_e$). It receives the full sequence (encoded visible tokens + mask tokens at masked positions) and produces per-patch reconstructions:

$$\hat{\mathbf{p}}_i = \mathbf{W}_{\text{out}} \cdot \mathbf{D}_i^{(L_d)} + \mathbf{b}_{\text{out}}$$

where $\hat{\mathbf{p}}_i \in \mathbb{R}^{5P}$ is the reconstructed patch.

### Fine-Tuning for Trading

After pre-training, the decoder is discarded and the encoder is used for downstream tasks:

1. **Trend Prediction**: Attach a classification head to predict up/down movement.
2. **Volatility Forecasting**: Attach a regression head for future volatility estimation.
3. **Anomaly Detection**: Use reconstruction error as an anomaly score -- high reconstruction error indicates unusual market behavior.

The fine-tuning loss for classification:

$$\mathcal{L}_{\text{fine-tune}} = -\sum_{c} y_c \log(\hat{y}_c)$$

## Implementation in Rust

Our Rust implementation provides production-grade performance for MAE-based trading systems. Key components include:

### Core Data Structures

The implementation uses `ndarray` for efficient matrix operations. The `MaskedAutoencoder` struct encapsulates the entire architecture:

- `PatchEmbedding`: Converts raw OHLCV patches into embeddings
- `MaskedAutoencoder`: Full MAE with encoder, decoder, masking logic, and training
- `TradingStrategy`: Wraps a pre-trained MAE for generating trading signals

### Random Masking Strategy

The masking function generates a random permutation of patch indices and selects the first $\lfloor r \cdot N \rfloor$ as masked positions. This ensures exactly the desired masking ratio each time.

### Bybit Integration

The implementation fetches real market data from Bybit's public API:

```rust
let url = format!(
    "https://api.bybit.com/v5/market/kline?category=linear&symbol={}&interval={}&limit={}",
    symbol, interval, limit
);
```

The response is parsed into OHLCV candles used for both pre-training and strategy evaluation.

### Training Loop

Pre-training iterates over the dataset with random masking each epoch:

1. Select random masking pattern
2. Extract visible patches and pass through encoder
3. Insert mask tokens and pass through decoder
4. Compute MSE loss on masked positions
5. Update weights via gradient descent

### Trading Signal Generation

After pre-training, the MAE encoder generates feature representations. The reconstruction error on recent data serves as an anomaly indicator:

- **Low reconstruction error**: Market behaves normally -- follow trend signals
- **High reconstruction error**: Unusual market conditions -- reduce position size or go to cash

## Practical Applications

### 1. Pre-Training on Historical Data

Use years of historical OHLCV data to pre-train the MAE. The model learns:
- Typical candlestick patterns
- Volume-price relationships
- Mean-reversion and momentum signatures
- Volatility clustering

### 2. Few-Shot Learning for New Markets

A MAE pre-trained on BTC/USDT can be fine-tuned with minimal labeled data for ETH/USDT or other pairs, transferring knowledge about general market microstructure.

### 3. Regime Detection

Tracking reconstruction error over time reveals market regime changes. A sudden spike in error indicates the market is behaving unlike historical norms.

### 4. Data Augmentation

The MAE decoder can generate synthetic market scenarios by varying the mask positions and reconstructions, useful for stress testing trading strategies.

## Why High Masking Ratios Work for Finance

In financial time series, adjacent patches are highly correlated (prices are continuous, volatility clusters). With low masking ratios (e.g., 25%), the model can trivially interpolate. At 75% masking, the model must:

1. Understand the underlying volatility regime
2. Infer trend direction from sparse observations
3. Model the joint distribution of OHLCV features
4. Capture long-range dependencies across the visible patches

This creates a challenging pretext task that yields rich representations.

## Preprocessing for Financial MAE

Financial data requires careful normalization:

1. **Log returns**: Convert prices to log returns for stationarity: $r_t = \ln(P_t / P_{t-1})$
2. **Volume normalization**: Apply log transform and z-score normalization
3. **Per-patch normalization**: Subtract patch mean and divide by patch standard deviation before computing reconstruction loss
4. **Handling missing data**: Mark truly missing data separately from artificial masking

## Advanced Masking Strategies for Finance

Beyond uniform random masking, financial-specific strategies include:

- **Block masking**: Mask contiguous time blocks to force longer-range prediction
- **Feature masking**: Mask specific features (e.g., all volume data) to learn cross-feature dependencies
- **Regime-aware masking**: Increase masking during volatile periods where reconstruction is harder
- **Multi-scale masking**: Apply different masking ratios at different temporal scales

## Transfer to Downstream Tasks

### Regime Detection

The pretrained encoder captures latent market states. Fine-tuning for regime detection:

1. Freeze the MAE encoder (or use low learning rate)
2. Add a classification head: $\hat{y} = \text{softmax}(W_c \cdot h_{\text{[CLS]}} + b_c)$
3. Train on labeled regime data (bull/bear/sideways)

The pretrained representations typically require 10x fewer labeled samples than training from scratch.

### Anomaly Detection

The reconstruction error itself serves as an anomaly score:

$$\text{anomaly\_score}(X) = \frac{1}{N} \sum_{i=1}^{N} \|x_i - \hat{x}_i\|_2^2$$

Anomalous market conditions (flash crashes, circuit breakers, extreme events) produce high reconstruction errors because they deviate from learned normal patterns. This provides a natural, unsupervised anomaly detector.

### Return Prediction

Fine-tune the encoder for next-window return prediction:

1. Use the encoder's output representation
2. Add a regression head for predicting returns over the next $k$ periods
3. The pretrained features capture market dynamics that improve prediction accuracy

### Risk Assessment

The decoder's uncertainty in reconstruction correlates with market uncertainty:
- Patches that are consistently hard to reconstruct indicate unpredictable market regimes
- This provides a data-driven volatility/risk indicator

## Performance Considerations

| Aspect | Detail |
|--------|--------|
| Pre-training data | 10,000+ candles recommended |
| Masking ratio | 75% (optimal for financial data) |
| Patch size | 4-8 candles per patch |
| Encoder depth | 4-6 transformer blocks |
| Decoder depth | 2 blocks (lightweight) |
| Embedding dimension | 64-128 |
| Learning rate | 1e-4 with cosine schedule |
| Batch size | 32-64 |

## Risk Management Integration

The MAE reconstruction error provides a natural uncertainty measure:

$$\text{uncertainty}_t = \frac{1}{|\mathcal{M}|} \sum_{i \in \mathcal{M}} \| x_i^{(t)} - \hat{x}_i^{(t)} \|^2$$

This can be incorporated into position sizing:

$$\text{position\_size}_t = \text{base\_size} \times \frac{\sigma_{\text{target}}}{\sigma_t + \epsilon} \times \mathbb{1}[\text{uncertainty}_t < \tau]$$

where $\tau$ is an uncertainty threshold above which no position is taken.

## Key Takeaways

1. **Masked Autoencoders** provide a powerful self-supervised pre-training framework that learns rich representations from unlabeled financial data by reconstructing randomly masked portions of the input.

2. **High masking ratios (75%)** are beneficial for financial time series -- they prevent the model from relying on trivial interpolation and force it to learn meaningful patterns.

3. **Asymmetric encoder-decoder design** -- a deep encoder with a shallow decoder -- yields efficient pre-training where the encoder captures the most important features.

4. **Reconstruction error as anomaly score** provides a built-in uncertainty measure for risk management, enabling the strategy to reduce exposure during unusual market conditions.

5. **Transfer learning** is a key benefit: a MAE pre-trained on one instrument can be fine-tuned with minimal data for new instruments, addressing the perennial problem of limited labeled data in finance.

6. **Patch-based tokenization** of OHLCV data captures local temporal structure while enabling the transformer architecture to model long-range dependencies across the sequence.

7. **Production Rust implementation** delivers the performance needed for real-time trading applications, with efficient matrix operations via `ndarray` and live data integration through Bybit's API.

## References

1. He, K., Chen, X., Xie, S., Li, Y., Dollar, P., & Girshick, R. (2021). *Masked Autoencoders Are Scalable Vision Learners*. arXiv:2111.06377.
2. Devlin, J., et al. (2018). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. arXiv:1810.04805.
3. Vaswani, A., et al. (2017). *Attention Is All You Need*. NeurIPS 2017.
4. Assran, M., et al. (2022). *Masked Siamese Networks for Label-Efficient Learning*. ECCV 2022.
5. Dong, X., et al. (2022). *PatchTST: A Time Series is Worth 64 Words*. ICLR 2023.
6. Zerveas, G., et al. (2021). *A Transformer-based Framework for Multivariate Time Series Representation Learning*. KDD 2021.
