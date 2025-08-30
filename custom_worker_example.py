"""
Example of creating a custom worker following the patterns used in the VERL codebase.

This example demonstrates how to create a custom worker that can be integrated into the
VERL training pipeline, following the same patterns used for ActorRolloutRef and Critic workers.
"""

import os
import datetime
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass

import torch
import torch.distributed
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

# Import configuration classes (you may need to create these)
from verl.workers.config import ProfilerConfig

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@dataclass
class CustomWorkerConfig:
    """Configuration for the custom worker."""
    model_path: str
    strategy: str = "fsdp"  # or "megatron", "vllm", etc.
    profiler: Optional[Dict] = None
    custom_param1: str = "default_value"
    custom_param2: int = 42


class CustomWorker(Worker, DistProfilerExtension):
    """
    A custom worker that follows the same patterns as ActorRolloutRef and Critic workers.
    
    This worker demonstrates:
    1. Proper inheritance from Worker and DistProfilerExtension
    2. Distributed training setup
    3. Model initialization
    4. Method registration with dispatch modes
    5. Integration with the VERL training pipeline
    """
    
    def __init__(self, config: CustomWorkerConfig, role: str = "custom", **kwargs):
        """Initialize the custom worker.
        
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
                timeout=datetime.timedelta(seconds=600),
                init_method=os.environ.get("DIST_INIT_METHOD", None),
            )
        
        # Set up device mesh for distributed training
        world_size = torch.distributed.get_world_size()
        self.device_mesh = self._create_device_mesh(world_size)
        
        # Register dispatch/collect info for distributed operations
        self._register_dispatch_collect_info(
            mesh_name="custom", 
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
        self.optimizer = None
        self.custom_state = {}
        
        # You can add any custom initialization logic here
        logger.info(f"Initialized CustomWorker with role: {self.role}")
    
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        """Initialize the model and optimizer.
        
        This method is called by the training pipeline to set up the model.
        The @register decorator ensures proper distributed execution.
        """
        logger.info("Initializing custom worker model...")
        
        # Import external libraries if specified
        # import_external_libs(self.config.get("external_lib", None))
        
        # Create your model here
        self.model = self._create_model()
        
        # Create optimizer if needed
        self.optimizer = self._create_optimizer()
        
        # Log memory usage
        log_gpu_memory_usage("After custom worker model initialization", logger=logger)
        
        logger.info("Custom worker model initialization completed")
    
    def _create_model(self):
        """Create the model instance.
        
        Returns:
            The initialized model
        """
        # This is where you would create your specific model
        # Example:
        # from your_model import YourModel
        # model = YourModel(config=self.config)
        
        # For demonstration, we'll create a simple placeholder
        class PlaceholderModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 1)
            
            def forward(self, x):
                return self.linear(x)
        
        model = PlaceholderModel()
        
        # Move model to appropriate device
        device = torch.device(f"cuda:{self.rank}" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        return model
    
    def _create_optimizer(self):
        """Create the optimizer instance.
        
        Returns:
            The initialized optimizer
        """
        if self.model is None:
            return None
        
        # Create optimizer for your model
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        return optimizer
    
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def custom_forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """Custom forward pass method.
        
        Args:
            input_data: Input tensor for the model
            
        Returns:
            Model output
        """
        if self.model is None:
            raise RuntimeError("Model not initialized. Call init_model() first.")
        
        # Perform forward pass
        with torch.no_grad():
            output = self.model(input_data)
        
        return output
    
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def custom_training_step(self, batch_data: Dict[str, Any]) -> Dict[str, Any]:
        """Custom training step method.
        
        Args:
            batch_data: Batch of training data
            
        Returns:
            Training metrics and loss
        """
        if self.model is None or self.optimizer is None:
            raise RuntimeError("Model or optimizer not initialized. Call init_model() first.")
        
        # Set model to training mode
        self.model.train()
        
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Forward pass
        inputs = batch_data.get("inputs", torch.randn(32, 10))
        targets = batch_data.get("targets", torch.randn(32, 1))
        
        outputs = self.model(inputs)
        loss = torch.nn.functional.mse_loss(outputs, targets)
        
        # Backward pass
        loss.backward()
        
        # Optimizer step
        self.optimizer.step()
        
        # Return metrics
        return {
            "loss": loss.item(),
            "outputs": outputs.detach(),
            "custom_metric": self.config.custom_param2
        }
    
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, path: str, step: int):
        """Save checkpoint for the custom worker.
        
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
        
        # Save optimizer state if available
        if self.optimizer is not None:
            optimizer_path = os.path.join(path, "optimizer.pt")
            torch.save(self.optimizer.state_dict(), optimizer_path)
        
        # Save custom state
        custom_state_path = os.path.join(path, "custom_state.pt")
        torch.save(self.custom_state, custom_state_path)
        
        logger.info(f"Saved checkpoint to {path} at step {step}")
    
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, path: str):
        """Load checkpoint for the custom worker.
        
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
        
        # Load optimizer state if available
        if self.optimizer is not None:
            optimizer_path = os.path.join(path, "optimizer.pt")
            if os.path.exists(optimizer_path):
                self.optimizer.load_state_dict(torch.load(optimizer_path))
                logger.info(f"Loaded optimizer checkpoint from {optimizer_path}")
        
        # Load custom state
        custom_state_path = os.path.join(path, "custom_state.pt")
        if os.path.exists(custom_state_path):
            self.custom_state = torch.load(custom_state_path)
            logger.info(f"Loaded custom state from {custom_state_path}")
    
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def get_worker_info(self) -> Dict[str, Any]:
        """Get information about the worker.
        
        Returns:
            Dictionary containing worker information
        """
        return {
            "role": self.role,
            "rank": self.rank,
            "world_size": self.world_size,
            "strategy": self.config.strategy,
            "custom_param1": self.config.custom_param1,
            "custom_param2": self.config.custom_param2,
            "has_model": self.model is not None,
            "has_optimizer": self.optimizer is not None
        }


# Example of how to integrate the custom worker into the training pipeline
def create_custom_worker_integration_example():
    """
    Example showing how to integrate the custom worker into the VERL training pipeline.
    
    This follows the same pattern used for ActorRolloutRef and Critic workers.
    """
    
    # 1. Define the custom worker configuration
    custom_config = CustomWorkerConfig(
        model_path="/path/to/your/model",
        strategy="fsdp",
        custom_param1="example_value",
        custom_param2=100
    )
    
    # 2. Create the custom worker class with Ray remote
    import ray
    from verl.single_controller.ray.base import RayClassWithInitArgs
    from verl.trainer.ppo.ray_trainer import Role, ResourcePoolManager
    
    # Create a custom role (you can extend the Role enum)
    class ExtendedRole(Role):
        CustomWorker = 7  # Add your custom role
    
    # 3. Set up resource pool manager
    resource_pool_spec = {
        "global_pool": [4] * 2,  # 4 GPUs per node, 2 nodes
    }
    
    mapping = {
        ExtendedRole.CustomWorker: "global_pool",
    }
    
    resource_pool_manager = ResourcePoolManager(
        resource_pool_spec=resource_pool_spec, 
        mapping=mapping
    )
    
    # 4. Create role-worker mapping
    role_worker_mapping = {
        ExtendedRole.CustomWorker: ray.remote(CustomWorker),
    }
    
    # 5. Create worker configuration
    custom_worker_cls = RayClassWithInitArgs(
        cls=role_worker_mapping[ExtendedRole.CustomWorker],
        config=custom_config,
        role="custom"
    )
    
    # 6. Set up resource pool and worker group
    resource_pool_manager.create_resource_pool()
    resource_pool = resource_pool_manager.get_resource_pool(ExtendedRole.CustomWorker)
    
    # 7. Create worker group (this would be done in the trainer)
    # from verl.single_controller.ray.base import create_colocated_worker_cls, RayWorkerGroup
    # 
    # resource_pool_to_cls = {resource_pool: {"custom": custom_worker_cls}}
    # 
    # for pool, class_dict in resource_pool_to_cls.items():
    #     worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
    #     wg = RayWorkerGroup(resource_pool=pool, ray_cls_with_init=worker_dict_cls)
    #     custom_wg = wg.spawn(prefix_set=class_dict.keys())
    #     custom_wg["custom"].init_model()
    
    return custom_worker_cls


if __name__ == "__main__":
    # Example usage
    print("Custom Worker Example")
    print("====================")
    
    # Create a simple configuration
    config = CustomWorkerConfig(
        model_path="/tmp/model",
        strategy="fsdp",
        custom_param1="test_value",
        custom_param2=42
    )
    
    # Create worker instance (for demonstration, not in distributed mode)
    worker = CustomWorker(config, role="demo")
    
    # Initialize model
    worker.init_model()
    
    # Get worker info
    info = worker.get_worker_info()
    print(f"Worker info: {info}")
    
    # Example forward pass
    dummy_input = torch.randn(5, 10)
    output = worker.custom_forward(dummy_input)
    print(f"Forward pass output shape: {output.shape}")
    
    # Example training step
    batch_data = {
        "inputs": torch.randn(16, 10),
        "targets": torch.randn(16, 1)
    }
    metrics = worker.custom_training_step(batch_data)
    print(f"Training metrics: {metrics}")
    
    print("\nCustom worker example completed successfully!")
