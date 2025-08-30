"""
TranslatorWorker - A custom worker for policy updates following VERL patterns.

This worker demonstrates how to create a concrete worker that can load models
from local paths or Hugging Face, following the same patterns used by
ActorRolloutRef and Critic workers in the VERL codebase.
"""

import os
import datetime
import logging
from typing import Optional, Dict, Any, Union
from dataclasses import dataclass

import torch
import torch.distributed
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

# Import VERL base classes and utilities
from verl.single_controller.base.worker import Worker
from verl.utils.device import get_device_name, get_nccl_backend
from verl.utils.profiler import DistProfilerExtension, DistProfiler
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.decorator import register, Dispatch
from verl.utils.fs import import_external_libs
from verl.utils.memory import log_gpu_memory_usage
from verl.utils.seed import set_random_seed

# Import configuration classes
from verl.workers.config import ProfilerConfig

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@dataclass
class TranslatorWorkerConfig:
    """Configuration for the translator worker."""
    model: Dict[str, Any]  # Model configuration including path
    strategy: str = "fsdp"  # or "megatron", "vllm", etc.
    profiler: Optional[Dict] = None
    use_fused_kernels: bool = False
    use_remove_padding: bool = False
    trust_remote_code: bool = False
    external_lib: Optional[str] = None
    
    # Training configuration
    optim: Optional[Dict[str, Any]] = None  # Optimizer configuration


class TranslatorWorker(Worker, DistProfilerExtension):
    """
    A translator worker that follows the same patterns as ActorRolloutRef and Critic workers.
    
    This worker demonstrates:
    1. Proper inheritance from Worker and DistProfilerExtension
    2. Distributed training setup
    3. Model loading from local path or Hugging Face
    4. Method registration with dispatch modes
    5. Integration with the VERL training pipeline
    6. Policy update capabilities
    """
    
    def __init__(self, config: TranslatorWorkerConfig, role: str = "translator", **kwargs):
        """Initialize the translator worker.
        
        Args:
            config: Configuration object for the worker
            role: Role identifier for the worker
            **kwargs: Additional keyword arguments
        """
        # Initialize base Worker class
        Worker.__init__(self)
        
        # Store configuration
        self.config = config
        self.role = role
        
        # Initialize distributed training if not already done
        if not torch.distributed.is_initialized():
            rank = int(os.environ.get("RANK", 0))
            world_size = int(os.environ.get("WORLD_SIZE", 1))
            torch.distributed.init_process_group(
                backend=f"cpu:gloo,{get_device_name()}:{get_nccl_backend()}",
                rank=rank,
                world_size=world_size,
                timeout=datetime.timedelta(seconds=self.config.get("nccl_timeout", 600)),
                init_method=os.environ.get("DIST_INIT_METHOD", None),
            )
        
        # Set up device mesh for distributed training
        world_size = torch.distributed.get_world_size()
        self.device_mesh = self._create_device_mesh(world_size)
        
        # Register dispatch/collect info for distributed operations
        self._register_dispatch_collect_info(
            mesh_name="translator", 
            dp_rank=self.rank, 
            is_collect=True
        )
        
        # Set random seed for reproducibility
        set_random_seed(seed=42)  # You can make this configurable
        
        # Initialize profiler if configured
        self._init_profiler()
        
        # Initialize worker-specific attributes
        self._init_worker_attributes()
    
    def _create_device_mesh(self, world_size: int):
        """Create device mesh for distributed training.
        
        Args:
            world_size: Total number of processes
            
        Returns:
            Device mesh for distributed operations
        """
        # This is a simplified example - you may need more sophisticated device mesh creation
        # based on your specific distributed training strategy
        if self.config.strategy == "fsdp":
            # For FSDP, you might create a simple device mesh
            from torch.distributed.device_mesh import init_device_mesh
            return init_device_mesh(
                get_device_name(), 
                mesh_shape=(world_size,), 
                mesh_dim_names=["dp"]
            )
        else:
            # For other strategies, you might need different device mesh setup
            return None
    
    def _init_profiler(self):
        """Initialize the profiler for performance monitoring."""
        if self.config.profiler:
            profiler_config = omega_conf_to_dataclass(
                self.config.profiler, 
                dataclass_type=ProfilerConfig
            )
            
            tool_config = None
            if self.config.profiler.get("tool") in ["npu", "nsys", "torch"]:
                tool_config = omega_conf_to_dataclass(
                    self.config.profiler.get("tool_config", {}).get(
                        self.config.profiler.get("tool")
                    )
                )
            
            DistProfilerExtension.__init__(
                self, 
                DistProfiler(rank=self.rank, config=profiler_config, tool_config=tool_config)
            )
    
    def _init_worker_attributes(self):
        """Initialize worker-specific attributes and state."""
        self.model = None
        self.tokenizer = None
        
        # Training components
        self.optimizer = None
        self.lr_scheduler = None
        
        # You can add any custom initialization logic here
        logger.info(f"Initialized TranslatorWorker with role: {self.role}")
    
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        """Initialize the model and tokenizer.
        
        This method is called by the training pipeline to set up the model.
        The @register decorator ensures proper distributed execution.
        """
        logger.info("Initializing translator worker model...")
        
        # Import external libraries if specified
        import_external_libs(self.config.external_lib)
        
        # Create your model here
        self.model, self.tokenizer = self._create_model_and_tokenizer()
        
        # Log memory usage
        log_gpu_memory_usage("After translator worker model initialization", logger=logger)
        
        # Initialize training components
        self._build_model_optimizer()
        
        logger.info("Translator worker model initialization completed")
    
    def _create_model_and_tokenizer(self):
        """Create the model and tokenizer instances.
        
        Returns:
            Tuple of (model, tokenizer)
        """
        model_path = self.config.model.get("path")
        if not model_path:
            raise ValueError("Model path must be specified in config.model.path")
        
        # Check if it's a local path or Hugging Face model
        if os.path.exists(model_path):
            logger.info(f"Loading model from local path: {model_path}")
            model, tokenizer = self._load_from_local_path(model_path)
        else:
            logger.info(f"Loading model from Hugging Face: {model_path}")
            model, tokenizer = self._load_from_huggingface(model_path)
        
        # Move model to appropriate device
        device = torch.device(f"cuda:{self.rank}" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        return model, tokenizer
    
    def _load_from_local_path(self, model_path: str):
        """Load model and tokenizer from local path.
        
        Args:
            model_path: Path to the local model directory
            
        Returns:
            Tuple of (model, tokenizer)
        """
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=self.config.trust_remote_code
            )
            
            # Load model
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=self.config.trust_remote_code,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            
            return model, tokenizer
            
        except ImportError:
            logger.warning("Transformers not available, trying to load with torch.load")
            # Fallback to torch.load for custom models
            model = torch.load(os.path.join(model_path, "model.pt"))
            tokenizer = None  # You might need to implement custom tokenizer loading
            
            return model, tokenizer
    
    def _load_from_huggingface(self, model_name: str):
        """Load model and tokenizer from Hugging Face.
        
        Args:
            model_name: Hugging Face model name
            
        Returns:
            Tuple of (model, tokenizer)
        """
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=self.config.trust_remote_code
            )
            
            # Load model
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=self.config.trust_remote_code,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            
            return model, tokenizer
            
        except ImportError:
            raise ImportError("Transformers library is required to load Hugging Face models")
    

    
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model.
        
        Returns:
            Dictionary containing model information
        """
        if self.model is None:
            return {"error": "Model not initialized"}
        
        # Get model information
        model_info = {
            "model_type": type(self.model).__name__,
            "num_parameters": sum(p.numel() for p in self.model.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            "device": next(self.model.parameters()).device,
            "dtype": next(self.model.parameters()).dtype,
            "has_tokenizer": self.tokenizer is not None,
            "role": self.role,
            "rank": self.rank,
            "world_size": self.world_size
        }
        
        # Add tokenizer info if available
        if self.tokenizer is not None:
            model_info.update({
                "vocab_size": self.tokenizer.vocab_size,
                "model_max_length": getattr(self.tokenizer, 'model_max_length', 'unknown'),
                "pad_token": self.tokenizer.pad_token,
                "eos_token": self.tokenizer.eos_token
            })
        
        return model_info
    
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, path: str, step: int):
        """Save checkpoint for the translator worker.
        
        Args:
            path: Path to save the checkpoint
            step: Current training step
        """
        if self.model is None:
            logger.warning("No model to save")
            return
        
        # Create checkpoint directory
        os.makedirs(path, exist_ok=True)
        
        # Save model state
        model_path = os.path.join(path, "model.pt")
        torch.save(self.model.state_dict(), model_path)
        
        # Save tokenizer if available
        if self.tokenizer is not None:
            tokenizer_path = os.path.join(path, "tokenizer")
            self.tokenizer.save_pretrained(tokenizer_path)
        
        # Save configuration
        config_path = os.path.join(path, "config.pt")
        torch.save(self.config, config_path)
        
        logger.info(f"Saved checkpoint to {path} at step {step}")
    
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, path: str):
        """Load checkpoint for the translator worker.
        
        Args:
            path: Path to load the checkpoint from
        """
        if self.model is None:
            logger.warning("No model to load checkpoint into")
            return
        
        # Load model state
        model_path = os.path.join(path, "model.pt")
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path))
            logger.info(f"Loaded model checkpoint from {model_path}")
        
        # Load tokenizer if available
        tokenizer_path = os.path.join(path, "tokenizer")
        if os.path.exists(tokenizer_path) and self.tokenizer is not None:
            try:
                from transformers import AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
                logger.info(f"Loaded tokenizer from {tokenizer_path}")
            except Exception as e:
                logger.warning(f"Failed to load tokenizer: {e}")
        
        # Load configuration
        config_path = os.path.join(path, "config.pt")
        if os.path.exists(config_path):
            loaded_config = torch.load(config_path)
            logger.info(f"Loaded configuration from {config_path}")
    

    
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def health_check(self) -> Dict[str, Any]:
        """Perform a health check on the worker.
        
        Returns:
            Health status information
        """
        status = {
            "status": "healthy",
            "role": self.role,
            "rank": self.role,
            "model_loaded": self.model is not None,
            "tokenizer_loaded": self.tokenizer is not None,
            "device_available": torch.cuda.is_available() if torch.cuda.is_available() else "CPU only"
        }
        
        # Check for potential issues
        if self.model is None:
            status["status"] = "unhealthy"
            status["error"] = "Model not initialized"
        elif self.tokenizer is None:
            status["status"] = "warning"
            status["warning"] = "Tokenizer not loaded"
        
        return status
    
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_log_prob(self, input_sequences: torch.Tensor, output_sequences: torch.Tensor, 
                        attention_mask: Optional[torch.Tensor] = None, 
                        temperature: float = 1.0) -> Dict[str, Any]:
        """Compute log probabilities of output sequences given input sequences.
        
        This method follows the same pattern as ActorRolloutRef worker's compute_log_prob method.
        It calculates the log probability of each token in the output sequence given the input sequence.
        
        Args:
            input_sequences: Input token sequences of shape [batch_size, input_length]
            output_sequences: Output token sequences of shape [batch_size, output_length]
            attention_mask: Optional attention mask for input sequences
            temperature: Temperature for logits scaling (default: 1.0)
            
        Returns:
            Dictionary containing:
                - log_probs: Log probabilities of shape [batch_size, output_length]
                - entropys: Entropy values of shape [batch_size, output_length] (if calculate_entropy=True)
                - metadata: Additional information about the computation
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model or tokenizer not initialized. Call init_model() first.")
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Move tensors to the same device as model
        device = next(self.model.parameters()).device
        input_sequences = input_sequences.to(device)
        output_sequences = output_sequences.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        
        # Concatenate input and output sequences for the model
        # The model expects the full sequence: [input + output]
        full_sequences = torch.cat([input_sequences, output_sequences], dim=1)
        
        # Create attention mask for full sequence if not provided
        if attention_mask is None:
            input_length = input_sequences.size(1)
            output_length = output_sequences.size(1)
            full_length = input_length + output_length
            
            # Create attention mask: 1 for input tokens, 1 for output tokens
            attention_mask = torch.ones(full_sequences.size(0), full_length, device=device, dtype=torch.long)
        
        # Create position IDs
        batch_size, seq_length = full_sequences.size()
        position_ids = torch.arange(seq_length, device=device).unsqueeze(0).expand(batch_size, -1)
        
        # Forward pass through the model
        with torch.no_grad():
            model_outputs = self.model(
                input_ids=full_sequences,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False
            )
        
        # Extract logits for the output portion
        logits = model_outputs.logits  # [batch_size, seq_length, vocab_size]
        
        # Get logits for the output sequence part (after input)
        input_length = input_sequences.size(1)
        output_logits = logits[:, input_length-1:-1, :]  # [batch_size, output_length, vocab_size]
        
        # Scale logits by temperature
        output_logits = output_logits / temperature
        
        # Calculate log probabilities using the same method as ActorRolloutRef worker
        log_probs = self._compute_log_probs_from_logits(output_logits, output_sequences)
        
        # Calculate entropy if requested
        entropys = self._compute_entropy_from_logits(output_logits)
        
        # Prepare output
        result = {
            "log_probs": log_probs,
            "entropys": entropys,
            "metadata": {
                "input_length": input_sequences.size(1),
                "output_length": output_sequences.size(1),
                "batch_size": batch_size,
                "temperature": temperature,
                "device": str(device),
                "worker_rank": self.rank
            }
        }
        
        return result
    
    def _compute_log_probs_from_logits(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute log probabilities from logits using the same method as ActorRolloutRef worker.
        
        Args:
            logits: Model logits of shape [batch_size, seq_length, vocab_size]
            labels: Target labels of shape [batch_size, seq_length]
            
        Returns:
            Log probabilities of shape [batch_size, seq_length]
        """
        # This is the same implementation as verl.utils.torch_functional.logprobs_from_logits_v2
        # which is used by the ActorRolloutRef worker
        
        if logits.dtype in [torch.float32, torch.float64]:
            # Memory efficient approach for float32/float64
            logits_labels = torch.gather(logits, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
            
            # Loop to reduce peak memory consumption
            logsumexp_values = torch.stack([
                torch.logsumexp(logit, dim=-1) for logit in logits
            ])
            
            # log_softmax(x_i) = x_i - logsumexp(x)
            logprobs_labels = logits_labels - logsumexp_values
        else:
            # For bfloat16, use the slightly less efficient but stable approach
            logprobs_labels = []
            for row_logits, row_labels in zip(logits, labels, strict=True):
                row_logprobs = torch.nn.functional.log_softmax(row_logits, dim=-1)
                row_logprobs_labels = row_logprobs.gather(dim=-1, index=row_labels.unsqueeze(-1)).squeeze(-1)
                logprobs_labels.append(row_logprobs_labels)
            logprobs_labels = torch.stack(logprobs_labels)
        
        return logprobs_labels
    
    def _compute_entropy_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute entropy from logits using the same method as ActorRolloutRef worker.
        
        Args:
            logits: Model logits of shape [batch_size, seq_length, vocab_size]
            
        Returns:
            Entropy values of shape [batch_size, seq_length]
        """
        # This is the same implementation as verl.utils.torch_functional.entropy_from_logits
        # which is used by the ActorRolloutRef worker
        
        # Calculate softmax probabilities
        pd = torch.nn.functional.softmax(logits, dim=-1)
        
        # Calculate entropy: H(p) = -sum(p * log(p))
        # Using the identity: H(p) = logsumexp(logits) - sum(p * logits)
        entropy = torch.logsumexp(logits, dim=-1) - torch.sum(pd * logits, dim=-1)
        
        return entropy
    

    
    def _build_model_optimizer(self):
        """Build model and optimizer following the same pattern as FSDP SFT trainer."""
        if self.model is None:
            logger.warning("Model not initialized, cannot build optimizer")
            return
        
        logger.info("Building model and optimizer...")
        
        # Get optimizer configuration with defaults
        optim_config = self.config.optim or {}
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=optim_config.get("lr", 1e-5),
            betas=optim_config.get("betas", (0.9, 0.999)),
            weight_decay=optim_config.get("weight_decay", 0.01),
            eps=optim_config.get("eps", 1e-8)
        )
        logger.info(f"Initialized AdamW optimizer with lr={optim_config.get('lr', 1e-5)}")
        
        # Initialize learning rate scheduler if configured
        if optim_config.get("lr_scheduler") == "cosine":
            total_steps = optim_config.get("total_steps", 1000)
            warmup_steps = optim_config.get("warmup_steps", 100)
            
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=total_steps - warmup_steps,
                T_mult=1
            )
            logger.info(f"Initialized cosine annealing scheduler with {total_steps} total steps, {warmup_steps} warmup")
        
        elif optim_config.get("lr_scheduler") == "linear":
            total_steps = optim_config.get("total_steps", 1000)
            warmup_steps = optim_config.get("warmup_steps", 100)
            
            def lr_lambda(step):
                if step < warmup_steps:
                    return float(step) / float(max(1, warmup_steps))
                return max(0.0, float(total_steps - step) / float(max(1, total_steps - warmup_steps)))
            
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
            logger.info(f"Initialized linear scheduler with {total_steps} total steps, {warmup_steps} warmup")
        
        # Enable gradient checkpointing if configured
        if optim_config.get("gradient_checkpointing", False):
            self.model.gradient_checkpointing_enable()
            logger.info("Enabled gradient checkpointing")
        
        # Enable input requires grad if using LoRA or other parameter-efficient methods
        if optim_config.get("enable_input_requires_grad", False):
            self.model.enable_input_require_grads()
            logger.info("Enabled input requires grad")
        
        logger.info("Model and optimizer built successfully")
    

    
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_policy(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Update the policy model using the provided data.
        
        This method implements policy updates by computing loss and performing
        optimization steps. It follows the same pattern as the FSDP SFT trainer.
        
        Args:
            data: Dictionary containing training data:
                - input_ids: Input token sequences [batch_size, seq_length]
                - attention_mask: Attention mask for sequences
                - position_ids: Position IDs for sequences
                - labels: Target labels for loss computation
                - loss_mask: Optional mask for loss computation
                
        Returns:
            Dictionary containing training metrics and loss information
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model or tokenizer not initialized. Call init_model() first.")
        
        if self.optimizer is None:
            raise RuntimeError("Optimizer not initialized. Call _build_model_optimizer() first.")
        
        # Set model to training mode
        self.model.train()
        
        # Extract data
        input_ids = data["input_ids"]
        attention_mask = data.get("attention_mask")
        position_ids = data.get("position_ids")
        labels = data["labels"]
        loss_mask = data.get("loss_mask")
        
        # Move tensors to the same device as model
        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        if position_ids is not None:
            position_ids = position_ids.to(device)
        labels = labels.to(device)
        if loss_mask is not None:
            loss_mask = loss_mask.to(device)
        
        # Create position IDs if not provided
        if position_ids is None:
            batch_size, seq_length = input_ids.size()
            position_ids = torch.arange(seq_length, device=device).unsqueeze(0).expand(batch_size, -1)
        
        # Forward pass through the model
        with torch.autocast(device_type=str(device), dtype=torch.bfloat16):
            model_outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False
            )
        
        # Extract logits and compute loss
        logits = model_outputs.logits  # [batch_size, seq_length, vocab_size]
        
        # For policy updates, we want to predict the next token given the current token
        # Shift logits and labels for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()  # [batch_size, seq_length-1, vocab_size]
        shift_labels = labels[..., 1:].contiguous()       # [batch_size, seq_length-1]
        
        # Flatten for loss computation
        batch_size, seq_length, vocab_size = shift_logits.size()
        shift_logits = shift_logits.view(-1, vocab_size)
        shift_labels = shift_labels.view(-1)
        
        # Compute cross-entropy loss
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(shift_logits, shift_labels)
        
        # Apply loss mask if provided
        if loss_mask is not None:
            # Adjust mask to match the shifted sequence length
            loss_mask = loss_mask[:, :-1].reshape(-1)
            loss_mask = loss_mask.to(loss.device)
            loss = loss * loss_mask
        
        # Compute average loss
        if loss_mask is not None:
            valid_tokens = torch.sum(loss_mask)
            if valid_tokens > 0:
                loss = torch.sum(loss) / valid_tokens
            else:
                loss = torch.mean(loss)
        else:
            loss = torch.mean(loss)
        
        # Backward pass and optimization
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping if configured
        grad_norm = None
        if hasattr(self.config.optim, "clip_grad_norm") and self.config.optim.clip_grad_norm:
            max_norm = self.config.optim.clip_grad_norm
            if hasattr(self.model, "clip_grad_norm_"):
                # For FSDP models
                grad_norm = self.model.clip_grad_norm_(max_norm=max_norm)
            else:
                # For regular models
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_norm)
            
            # Check if gradients are finite
            if not torch.isfinite(grad_norm):
                logger.warning(f"Gradient norm is not finite: {grad_norm}")
                self.optimizer.zero_grad()
                return {
                    "loss": float('inf'),
                    "grad_norm": float('inf'),
                    "status": "gradient_clipped"
                }
        
        # Optimizer step
        self.optimizer.step()
        
        # Learning rate scheduler step if available
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        
        # Set model back to evaluation mode
        self.model.eval()
        
        # Prepare return data
        result = {
            "loss": loss.item(),
            "grad_norm": grad_norm.item() if grad_norm is not None else None,
            "batch_size": batch_size,
            "seq_length": seq_length,
            "vocab_size": vocab_size,
            "device": str(device),
            "worker_rank": self.rank,
            "status": "success"
        }
        
        # Add learning rate if available
        if self.lr_scheduler is not None:
            result["learning_rate"] = self.lr_scheduler.get_last_lr()[0]
        
        return result
    
    def _init_sft_training(self):
        """Initialize SFT training components (optimizer, scheduler, etc.)."""
        if not self.config.sft or not self.config.sft.get("enabled", False):
            return
        
        logger.info("Initializing SFT training components...")
        
        # Get SFT configuration with defaults
        sft_config = self.config.sft
        
        # Initialize optimizer
        if self.model is not None:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=sft_config.get("learning_rate", 1e-5),
                betas=sft_config.get("betas", (0.9, 0.999)),
                weight_decay=sft_config.get("weight_decay", 0.01),
                eps=sft_config.get("eps", 1e-8)
            )
            logger.info(f"Initialized AdamW optimizer with lr={sft_config.get('learning_rate', 1e-5)}")
        else:
            logger.warning("Model not initialized, cannot create optimizer")
            return
        
        # Initialize learning rate scheduler if configured
        if sft_config.get("lr_scheduler") == "cosine":
            total_steps = sft_config.get("total_steps", 1000)
            warmup_steps = sft_config.get("warmup_steps", 100)
            
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=total_steps - warmup_steps,
                T_mult=1
            )
            logger.info(f"Initialized cosine annealing scheduler with {total_steps} total steps, {warmup_steps} warmup")
        
        elif sft_config.get("lr_scheduler") == "linear":
            total_steps = sft_config.get("total_steps", 1000)
            warmup_steps = sft_config.get("warmup_steps", 100)
            
            def lr_lambda(step):
                if step < warmup_steps:
                    return float(step) / float(max(1, warmup_steps))
                return max(0.0, float(total_steps - step) / float(max(1, total_steps - warmup_steps)))
            
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
            logger.info(f"Initialized linear scheduler with {total_steps} total steps, {warmup_steps} warmup")
        
        # Enable gradient checkpointing if configured
        if sft_config.get("gradient_checkpointing", False):
            self.model.gradient_checkpointing_enable()
            logger.info("Enabled gradient checkpointing for SFT training")
        
        # Enable input requires grad if using LoRA or other parameter-efficient methods
        if sft_config.get("enable_input_requires_grad", False):
            self.model.enable_input_require_grads()
            logger.info("Enabled input requires grad for SFT training")
        
        logger.info("SFT training components initialized successfully")
    
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_log_prob_from_text(self, input_texts: list[str], output_texts: list[str], 
                                 temperature: float = 1.0) -> Dict[str, Any]:
        """Compute log probabilities of output texts given input texts.
        
        This is a convenience method that handles tokenization and calls compute_log_prob.
        
        Args:
            input_texts: List of input text strings
            output_texts: List of output text strings (must have same length as input_texts)
            temperature: Temperature for logits scaling (default: 1.0)
            
        Returns:
            Dictionary containing log probabilities, entropy, and metadata
        """
        if len(input_texts) != len(output_texts):
            raise ValueError("Input texts and output texts must have the same length")
        
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not initialized. Call init_model() first.")
        
        # Tokenize input and output texts
        input_encodings = self.tokenizer(
            input_texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            add_special_tokens=True
        )
        
        output_encodings = self.tokenizer(
            output_texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            add_special_tokens=True
        )
        
        # Extract token IDs
        input_sequences = input_encodings["input_ids"]
        output_sequences = output_encodings["input_ids"]  # tokenizer returns input_ids for both
        attention_mask = input_encodings["attention_mask"]
        
        # Remove special tokens from output sequences (keep only the actual output tokens)
        # This assumes the output doesn't need special tokens for log probability calculation
        output_sequences = output_sequences[:, 1:-1]  # Remove BOS and EOS tokens
        
        # Call the main compute_log_prob method
        return self.compute_log_prob(
            input_sequences=input_sequences,
            output_sequences=output_sequences,
            attention_mask=attention_mask,
            temperature=temperature
        )
    
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_sft_from_text(self, input_texts: list[str], target_texts: list[str], 
                            loss_mask: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """Update the model using SFT with text inputs.
        
        This is a convenience method that handles tokenization and calls update_sft.
        
        Args:
            input_texts: List of input text strings
            target_texts: List of target text strings (must have same length as input_texts)
            loss_mask: Optional mask for loss computation
            
        Returns:
            Dictionary containing training metrics and loss information
        """
        if len(input_texts) != len(target_texts):
            raise ValueError("Input texts and target texts must have the same length")
        
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not initialized. Call init_model() first.")
        
        # Tokenize input and target texts
        input_encodings = self.tokenizer(
            input_texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            add_special_tokens=True
        )
        
        target_encodings = self.tokenizer(
            target_texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            add_special_tokens=True
        )
        
        # Extract token IDs
        input_ids = input_encodings["input_ids"]
        target_ids = target_encodings["input_ids"]
        attention_mask = input_encodings["attention_mask"]
        
        # Create labels (target_ids) and prepare for SFT
        # For SFT, we want to predict the target sequence given the input
        # We'll concatenate input and target for the full sequence
        full_sequences = torch.cat([input_ids, target_ids], dim=1)
        
        # Create attention mask for full sequence
        batch_size, full_length = full_sequences.size()
        full_attention_mask = torch.ones(batch_size, full_length, device=input_ids.device, dtype=torch.long)
        
        # Create position IDs
        position_ids = torch.arange(full_length, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        
        # Create loss mask (only compute loss on target tokens)
        if loss_mask is None:
            input_length = input_ids.size(1)
            target_length = target_ids.size(1)
            # Create mask: 0 for input tokens, 1 for target tokens
            loss_mask = torch.zeros(batch_size, full_length, device=input_ids.device, dtype=torch.long)
            loss_mask[:, input_length:] = 1
        
        # Prepare batch data for SFT update
        batch_data = {
            "input_ids": full_sequences,
            "attention_mask": full_attention_mask,
            "position_ids": position_ids,
            "labels": full_sequences,  # For SFT, labels are the same as input_ids
            "loss_mask": loss_mask
        }
        
        # Call the main SFT update method
        return self.update_sft(batch_data)


# Example of how to integrate the translator worker into the training pipeline
def create_translator_worker_integration_example():
    """
    Example showing how to integrate the translator worker into the VERL training pipeline.
    
    This follows the same pattern used for ActorRolloutRef and Critic workers.
    """
    
    # 1. Define the translator worker configuration
    translator_config = TranslatorWorkerConfig(
        model={
            "path": "Helsinki-NLP/opus-mt-en-fr",  # Hugging Face model
            # or "path": "/local/path/to/model" for local model
        },
        strategy="fsdp",
        trust_remote_code=False,
        translation_config={
            "max_length": 512,
            "num_beams": 4,
            "do_sample": False,
            "temperature": 1.0
        }
    )
    
    # 2. Create the translator worker class with Ray remote
    import ray
    from verl.single_controller.ray.base import RayClassWithInitArgs
    from verl.trainer.ppo.ray_trainer import Role, ResourcePoolManager
    
    # Create a custom role (you can extend the Role enum)
    class ExtendedRole(Role):
        Translator = 7  # Add your custom role
    
    # 3. Set up resource pool manager
    resource_pool_spec = {
        "global_pool": [4] * 2,  # 4 GPUs per node, 2 nodes
    }
    
    mapping = {
        ExtendedRole.Translator: "global_pool",
    }
    
    resource_pool_manager = ResourcePoolManager(
        resource_pool_spec=resource_pool_spec, 
        mapping=mapping
    )
    
    # 4. Create role-worker mapping
    role_worker_mapping = {
        ExtendedRole.Translator: ray.remote(TranslatorWorker),
    }
    
    # 5. Create worker configuration
    translator_worker_cls = RayClassWithInitArgs(
        cls=role_worker_mapping[ExtendedRole.Translator],
        config=translator_config,
        role="translator"
    )
    
    # 6. Set up resource pool and worker group
    resource_pool_manager.create_resource_pool()
    resource_pool = resource_pool_manager.get_resource_pool(ExtendedRole.Translator)
    
    # 7. Create worker group (this would be done in the trainer)
    # from verl.single_controller.ray.base import create_colocated_worker_cls, RayWorkerGroup
    # 
    # resource_pool_to_cls = {resource_pool: {"translator": translator_worker_cls}}
    # 
    # for pool, class_dict in resource_pool_to_cls.items():
    #     worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
    #     wg = RayWorkerGroup(resource_pool=pool, ray_cls_with_init=worker_dict_cls)
    #     translator_wg = wg.spawn(prefix_set=class_dict.keys())
    #     translator_wg["translator"].init_model()
    
    return translator_worker_cls


if __name__ == "__main__":
    # Example usage
    print("Translator Worker Example")
    print("=========================")
    
    # Create a simple configuration
    config = TranslatorWorkerConfig(
        model={
            "path": "Helsinki-NLP/opus-mt-en-fr"  # Example Hugging Face model
        },
        strategy="fsdp",
        trust_remote_code=False,
        translation_config={
            "max_length": 128,
            "num_beams": 2
        }
    )
    
    # Create worker instance (for demonstration, not in distributed mode)
    worker = TranslatorWorker(config, role="demo")
    
    # Initialize model
    worker.init_model()
    
    # Get model info
    info = worker.get_model_info()
    print(f"Model info: {info}")
    
    # Example translation
    input_texts = ["Hello, how are you?", "What is your name?"]
    translations = worker.translate(input_texts)
    print(f"Translations: {translations}")
    
    # Example log probability calculation
    print("\n--- Log Probability Calculation Example ---")
    input_texts = ["The weather is", "I like to"]
    output_texts = ["nice today", "read books"]
    
    try:
        log_prob_result = worker.compute_log_prob_from_text(input_texts, output_texts, temperature=1.0)
        print(f"Log probability result: {log_prob_result}")
        
        # Extract specific values
        log_probs = log_prob_result["log_probs"]
        entropys = log_prob_result["entropys"]
        metadata = log_prob_result["metadata"]
        
        print(f"Log probabilities shape: {log_probs.shape}")
        print(f"Entropy shape: {entropys.shape}")
        print(f"Average log probability per sequence:")
        for i in range(log_probs.size(0)):
            avg_log_prob = log_probs[i].mean().item()
            print(f"  Sequence {i}: {avg_log_prob:.4f}")
        
        print(f"Metadata: {metadata}")
        
    except Exception as e:
        print(f"Log probability calculation failed: {e}")
    
    # Health check
    health = worker.health_check()
    print(f"Health status: {health}")
    
    print("\nTranslator worker example completed successfully!")
