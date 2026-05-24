
from dataclasses import dataclass
import torch


@dataclass(frozen=True, slots=True, kw_only=True)
class GPTConfig():
    """
    Configuration for GPT model training and inference.
    All parameters are keyword-only and immutable after initialization.
    """
    
    # Model Architecture
    block_size     : int = 8
    n_layer        : int = 4
    n_head         : int = 4
    n_embd         : int = 64
    vocab_size     : int = 65
    attention_type : str = "single"
    
    # Training Data
    batch_size     : int = 32
    train_ratio    : float = 0.9

    # Optimization
    learning_rate  : float = 1e-3
    weight_decay   : float = 1e-4
    gradient_clip  : float = 1.0
    warmup_iters   : int = 0
    dropout        : float = 0.0
    
    # Training Loop
    max_iter       : int = 600000
    num_iter       : int = 100000
    eval_interval  : int = 5000
    eval_iters     : int = 200
    
    # Generation
    max_new_tokens : int = 1000
    temperature    : float = 1.0
    top_k          : int = None
    
    # Device & Reproducibility
    device         : torch.device = None  # Will be set in __post_init__
    seed           : int = 42
    
    def __post_init__(self):
        """
        Validates configuration parameters and sets up reproducibility.
        Raises ValueError if any parameter is invalid.
        """
        
        # ===== Device Setup =====
        if self.device is None:
            device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"Device string: {device_str}")
            object.__setattr__(self, 'device', torch.device(device_str))
        elif isinstance(self.device, str):
            object.__setattr__(self, 'device', torch.device(self.device))
        
        # ===== Architecture Validation =====
        if self.n_embd % self.n_head != 0:
            raise ValueError(
                f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head}). "
                f"Suggested: n_embd={self.n_head * (self.n_embd // self.n_head)}"
            )
        
        if self.n_layer <= 0:
            raise ValueError(f"n_layer must be positive, got {self.n_layer}")
        
        if self.n_head <= 0:
            raise ValueError(f"n_head must be positive, got {self.n_head}")
        
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {self.vocab_size}")

        if self.attention_type not in {"single", "multi"}:
            raise ValueError(
                "attention_type must be 'single' or 'multi', "
                f"got {self.attention_type}"
            )
        
        # ===== Training Parameter Validation =====
        if not (0 < self.train_ratio < 1):
            raise ValueError(f"train_ratio must be between 0 and 1, got {self.train_ratio}")
        
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        
        if self.block_size <= 0:
            raise ValueError(f"block_size must be positive, got {self.block_size}")
        
        # ===== Optimization Parameter Validation =====
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        
        if self.weight_decay < 0:
            raise ValueError(f"weight_decay must be non-negative, got {self.weight_decay}")
        
        if self.gradient_clip <= 0:
            raise ValueError(f"gradient_clip must be positive, got {self.gradient_clip}")
        
        if self.warmup_iters < 0:
            raise ValueError(f"warmup_iters must be non-negative, got {self.warmup_iters}")

        if not (0.0 <= self.dropout < 1.0):
            raise ValueError(f"dropout must be in [0.0, 1.0), got {self.dropout}")
        
        # ===== Training Loop Validation =====
        if self.eval_interval <= 0:
            raise ValueError(f"eval_interval must be positive, got {self.eval_interval}")
        
        if self.eval_iters <= 0:
            raise ValueError(f"eval_iters must be positive, got {self.eval_iters}")
        
        # ===== Generation Validation =====
        if self.max_new_tokens <= 0:
            raise ValueError(f"max_new_tokens must be positive, got {self.max_new_tokens}")
        
        if self.temperature <= 0:
            raise ValueError(f"temperature must be positive, got {self.temperature}")
        
        if self.top_k is not None and self.top_k <= 0:
            raise ValueError(f"top_k must be positive or None, got {self.top_k}")
        
        # ===== Device Validation =====
        device_type = self.device.type
        if device_type not in ['cuda', 'cpu']:
            raise ValueError(f"device must be 'cuda' or 'cpu', got {device_type}")
        
        # Adjust device if CUDA not available
        if device_type == 'cuda' and not torch.cuda.is_available():
            object.__setattr__(self, 'device', torch.device('cpu'))
            print("⚠️  CUDA not available, using CPU instead")
        
        # Set reproducibility seed
        torch.manual_seed(self.seed)
        if self.device == 'cuda':
            torch.cuda.manual_seed_all(self.seed)
        
        # ===== Summary Output =====
        print(f"GPTConfig validated successfully!")
        print(f"  Device: {self.device}")
        print(f"  Model: {self.n_layer} layers, {self.n_head} heads, {self.n_embd} embedding dim")
        print(f"  Head Dim: {self.head_size} | Vocab: {self.vocab_size}")
        print(f"  Training: batch_size={self.batch_size}, block_size={self.block_size}")
        print(f"  Estimated params: {self.num_params:,} ({self.estimated_model_size_mb:.2f} MB)")
    
    @property
    def head_size(self) -> int:
        """Dimension of each attention head."""
        return self.n_embd // self.n_head
    
    @property
    def num_params(self) -> int:
        """Rough estimate of total model parameters."""
        # Embeddings: token_embedding + position_embedding
        embed_params = (self.vocab_size + self.block_size) * self.n_embd
        
        # Transformer layers: each layer has attention + feedforward
        # Simplified estimate: ~12 * n_embd^2 per layer
        transformer_params = self.n_layer * 12 * (self.n_embd ** 2)
        
        # Output layer (LM head)
        head_params = self.n_embd * self.vocab_size
        
        total = embed_params + transformer_params + head_params
        return total
    
    @property
    def estimated_model_size_mb(self) -> float:
        """Rough memory footprint in MB (float32)."""
        # Each float32 parameter takes 4 bytes
        return (self.num_params * 4) / (1024 ** 2)
    
    def __repr__(self) -> str:
        """Custom string representation for better readability."""
        return (
            f"GPTConfig("
            f"block_size={self.block_size}, "
            f"n_layer={self.n_layer}, "
            f"n_head={self.n_head}, "
            f"n_embd={self.n_embd}, "
            f"vocab_size={self.vocab_size}, "
            f"batch_size={self.batch_size}, "
            f"lr={self.learning_rate})"
        )
        