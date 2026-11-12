// Generated from the native Lua registry. Do not edit.
// Each argument list preserves Lua positional/optional arguments; use Value::Null to skip one.
#[allow(non_snake_case)] pub mod Allocator {
    /// Native Mimir.Allocator.configure(cfg).
    pub fn r#configure(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Allocator.configure", args) }
    /// Native Mimir.Allocator.getStats().
    pub fn r#getStats(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Allocator.getStats", args) }
    /// Native Mimir.Allocator.get_stats().
    pub fn r#get_stats(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Allocator.get_stats", args) }
    /// Native Mimir.Allocator.printStats().
    pub fn r#printStats(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Allocator.printStats", args) }
    /// Native Mimir.Allocator.print_stats().
    pub fn r#print_stats(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Allocator.print_stats", args) }
}
#[allow(non_snake_case)] pub mod Architectures {
    /// Native Mimir.Architectures.available().
    pub fn r#available(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Architectures.available", args) }
    /// Native Mimir.Architectures.default_config(name).
    pub fn r#default_config(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Architectures.default_config", args) }
    /// Native Mimir.Architectures.dtypes().
    pub fn r#dtypes(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Architectures.dtypes", args) }
    /// Native Mimir.Architectures.info(name).
    pub fn r#info(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Architectures.info", args) }
}
#[allow(non_snake_case)] pub mod Benchmark {
    /// Native Mimir.Benchmark.now().
    pub fn r#now(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Benchmark.now", args) }
}
#[allow(non_snake_case)] pub mod Checkpoint {
    /// Native Mimir.Checkpoint.load(path, format, options).
    pub fn r#load(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Checkpoint.load", args) }
    /// Native Mimir.Checkpoint.save(path, format, options).
    pub fn r#save(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Checkpoint.save", args) }
}
#[allow(non_snake_case)] pub mod Database {
    /// Native Mimir.Database.load(dir, target_w, target_h, min_modalities).
    pub fn r#load(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Database.load", args) }
}
#[allow(non_snake_case)] pub mod Dataset {
    /// Native Mimir.Dataset.get(index, load_image).
    pub fn r#get(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Dataset.get", args) }
    /// Native Mimir.Dataset.load(dir, target_w, target_h, min_modalities, use_cache, cache_path, max_ram_mb, lazy_loading).
    pub fn r#load(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Dataset.load", args) }
    /// Native Mimir.Dataset.prefetch(indices, load_image).
    pub fn r#prefetch(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Dataset.prefetch", args) }
    /// Native Mimir.Dataset.prefetch_stats().
    pub fn r#prefetch_stats(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Dataset.prefetch_stats", args) }
    /// Native Mimir.Dataset.prepare_sequences(max_length).
    pub fn r#prepare_sequences(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Dataset.prepare_sequences", args) }
    /// Native Mimir.Dataset.release(index).
    pub fn r#release(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Dataset.release", args) }
}
#[allow(non_snake_case)] pub mod Distributed {
    /// Native Mimir.Distributed.barrier().
    pub fn r#barrier(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Distributed.barrier", args) }
    /// Native Mimir.Distributed.initialize().
    pub fn r#initialize(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Distributed.initialize", args) }
    /// Native Mimir.Distributed.is_main_process().
    pub fn r#is_main_process(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Distributed.is_main_process", args) }
    /// Native Mimir.Distributed.rank().
    pub fn r#rank(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Distributed.rank", args) }
    /// Native Mimir.Distributed.shutdown().
    pub fn r#shutdown(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Distributed.shutdown", args) }
    /// Native Mimir.Distributed.world_size().
    pub fn r#world_size(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Distributed.world_size", args) }
}
#[allow(non_snake_case)] pub mod Guard {
    /// Native Mimir.Guard.getStats().
    pub fn r#getStats(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Guard.getStats", args) }
    /// Native Mimir.Guard.get_stats().
    pub fn r#get_stats(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Guard.get_stats", args) }
    /// Native Mimir.Guard.printStats().
    pub fn r#printStats(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Guard.printStats", args) }
    /// Native Mimir.Guard.print_stats().
    pub fn r#print_stats(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Guard.print_stats", args) }
    /// Native Mimir.Guard.reset().
    pub fn r#reset(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Guard.reset", args) }
    /// Native Mimir.Guard.setLimit(limit_mb).
    pub fn r#setLimit(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Guard.setLimit", args) }
    /// Native Mimir.Guard.set_limit(limit_mb).
    pub fn r#set_limit(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Guard.set_limit", args) }
}
#[allow(non_snake_case)] pub mod Htop {
    /// Native Mimir.Htop.clear().
    pub fn r#clear(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Htop.clear", args) }
    /// Native Mimir.Htop.create(config).
    pub fn r#create(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Htop.create", args) }
    /// Native Mimir.Htop.enable(enabled).
    pub fn r#enable(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Htop.enable", args) }
    /// Native Mimir.Htop.render().
    pub fn r#render(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Htop.render", args) }
    /// Native Mimir.Htop.update(metrics, ...).
    pub fn r#update(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Htop.update", args) }
}
#[allow(non_snake_case)] pub mod IO {
    /// Native Mimir.IO.readImageRGBU8(path, target_w, target_h).
    pub fn r#readImageRGBU8(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("IO.readImageRGBU8", args) }
    /// Native Mimir.IO.read_image_rgb_u8(path, target_w, target_h).
    pub fn r#read_image_rgb_u8(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("IO.read_image_rgb_u8", args) }
    /// Native Mimir.IO.suppressStdoutLogs(enabled).
    pub fn r#suppressStdoutLogs(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("IO.suppressStdoutLogs", args) }
    /// Native Mimir.IO.suppress_stdout_logs(enabled).
    pub fn r#suppress_stdout_logs(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("IO.suppress_stdout_logs", args) }
}
#[allow(non_snake_case)] pub mod Layers {
    /// Native Mimir.Layers.Abs().
    pub fn r#Abs(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.Abs", args) }
    /// Native Mimir.Layers.AdaptiveAvgPool2d().
    pub fn r#AdaptiveAvgPool2d(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.AdaptiveAvgPool2d", args) }
    /// Native Mimir.Layers.Add().
    pub fn r#Add(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.Add", args) }
    /// Native Mimir.Layers.AlphaDropout().
    pub fn r#AlphaDropout(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.AlphaDropout", args) }
    /// Native Mimir.Layers.AvgPool1d().
    pub fn r#AvgPool1d(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.AvgPool1d", args) }
    /// Native Mimir.Layers.AvgPool2d().
    pub fn r#AvgPool2d(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.AvgPool2d", args) }
    /// Native Mimir.Layers.BatchMatMul().
    pub fn r#BatchMatMul(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.BatchMatMul", args) }
    /// Native Mimir.Layers.BatchNorm1d().
    pub fn r#BatchNorm1d(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.BatchNorm1d", args) }
    /// Native Mimir.Layers.BatchNorm2d().
    pub fn r#BatchNorm2d(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.BatchNorm2d", args) }
    /// Native Mimir.Layers.Bilinear().
    pub fn r#Bilinear(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.Bilinear", args) }
    /// Native Mimir.Layers.CausalDepthwiseConv1d().
    pub fn r#CausalDepthwiseConv1d(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.CausalDepthwiseConv1d", args) }
    /// Native Mimir.Layers.Chunk().
    pub fn r#Chunk(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.Chunk", args) }
    /// Native Mimir.Layers.Concat().
    pub fn r#Concat(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.Concat", args) }
    /// Native Mimir.Layers.Constant().
    pub fn r#Constant(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.Constant", args) }
    /// Native Mimir.Layers.Conv1d().
    pub fn r#Conv1d(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.Conv1d", args) }
    /// Native Mimir.Layers.Conv2d().
    pub fn r#Conv2d(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.Conv2d", args) }
    /// Native Mimir.Layers.Conv3d(...).
    pub fn r#Conv3d(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.Conv3d", args) }
    /// Native Mimir.Layers.Conv4d(...).
    pub fn r#Conv4d(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.Conv4d", args) }
    /// Native Mimir.Layers.ConvTranspose2d().
    pub fn r#ConvTranspose2d(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.ConvTranspose2d", args) }
    /// Native Mimir.Layers.Cos().
    pub fn r#Cos(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.Cos", args) }
    /// Native Mimir.Layers.CrossAttention().
    pub fn r#CrossAttention(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.CrossAttention", args) }
    /// Native Mimir.Layers.DepthwiseConv2d().
    pub fn r#DepthwiseConv2d(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.DepthwiseConv2d", args) }
    /// Native Mimir.Layers.Divide().
    pub fn r#Divide(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.Divide", args) }
    /// Native Mimir.Layers.Dropout().
    pub fn r#Dropout(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.Dropout", args) }
    /// Native Mimir.Layers.Dropout2d().
    pub fn r#Dropout2d(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.Dropout2d", args) }
    /// Native Mimir.Layers.Embedding().
    pub fn r#Embedding(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.Embedding", args) }
    /// Native Mimir.Layers.EmbeddingBag().
    pub fn r#EmbeddingBag(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.EmbeddingBag", args) }
    /// Native Mimir.Layers.Exp().
    pub fn r#Exp(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.Exp", args) }
    /// Native Mimir.Layers.Flatten().
    pub fn r#Flatten(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.Flatten", args) }
    /// Native Mimir.Layers.GEGLU().
    pub fn r#GEGLU(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.GEGLU", args) }
    /// Native Mimir.Layers.GELU().
    pub fn r#GELU(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.GELU", args) }
    /// Native Mimir.Layers.GRU().
    pub fn r#GRU(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.GRU", args) }
    /// Native Mimir.Layers.GlobalAvgPool2d().
    pub fn r#GlobalAvgPool2d(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.GlobalAvgPool2d", args) }
    /// Native Mimir.Layers.GroupNorm().
    pub fn r#GroupNorm(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.GroupNorm", args) }
    /// Native Mimir.Layers.HardSigmoid().
    pub fn r#HardSigmoid(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.HardSigmoid", args) }
    /// Native Mimir.Layers.HardSwish().
    pub fn r#HardSwish(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.HardSwish", args) }
    /// Native Mimir.Layers.Identity().
    pub fn r#Identity(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.Identity", args) }
    /// Native Mimir.Layers.InstanceNorm2d().
    pub fn r#InstanceNorm2d(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.InstanceNorm2d", args) }
    /// Native Mimir.Layers.LSTM().
    pub fn r#LSTM(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.LSTM", args) }
    /// Native Mimir.Layers.Lambda().
    pub fn r#Lambda(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.Lambda", args) }
    /// Native Mimir.Layers.LayerNorm().
    pub fn r#LayerNorm(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.LayerNorm", args) }
    /// Native Mimir.Layers.LeakyReLU().
    pub fn r#LeakyReLU(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.LeakyReLU", args) }
    /// Native Mimir.Layers.Linear().
    pub fn r#Linear(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.Linear", args) }
    /// Native Mimir.Layers.LogAbs().
    pub fn r#LogAbs(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.LogAbs", args) }
    /// Native Mimir.Layers.LogSoftmax().
    pub fn r#LogSoftmax(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.LogSoftmax", args) }
    /// Native Mimir.Layers.LowerTriangular().
    pub fn r#LowerTriangular(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.LowerTriangular", args) }
    /// Native Mimir.Layers.MatMul().
    pub fn r#MatMul(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.MatMul", args) }
    /// Native Mimir.Layers.MaxPool1d().
    pub fn r#MaxPool1d(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.MaxPool1d", args) }
    /// Native Mimir.Layers.MaxPool2d().
    pub fn r#MaxPool2d(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.MaxPool2d", args) }
    /// Native Mimir.Layers.Maximum().
    pub fn r#Maximum(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.Maximum", args) }
    /// Native Mimir.Layers.Minimum().
    pub fn r#Minimum(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.Minimum", args) }
    /// Native Mimir.Layers.Mish().
    pub fn r#Mish(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.Mish", args) }
    /// Native Mimir.Layers.MultiHeadAttention().
    pub fn r#MultiHeadAttention(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.MultiHeadAttention", args) }
    /// Native Mimir.Layers.Multiply().
    pub fn r#Multiply(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.Multiply", args) }
    /// Native Mimir.Layers.NMS().
    pub fn r#NMS(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.NMS", args) }
    /// Native Mimir.Layers.PatchEmbed(...).
    pub fn r#PatchEmbed(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.PatchEmbed", args) }
    /// Native Mimir.Layers.Permute().
    pub fn r#Permute(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.Permute", args) }
    /// Native Mimir.Layers.PixelShuffle().
    pub fn r#PixelShuffle(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.PixelShuffle", args) }
    /// Native Mimir.Layers.RMSNorm().
    pub fn r#RMSNorm(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.RMSNorm", args) }
    /// Native Mimir.Layers.RNN().
    pub fn r#RNN(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.RNN", args) }
    /// Native Mimir.Layers.ReLU().
    pub fn r#ReLU(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.ReLU", args) }
    /// Native Mimir.Layers.ReciprocalNoNan().
    pub fn r#ReciprocalNoNan(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.ReciprocalNoNan", args) }
    /// Native Mimir.Layers.ReduceMax().
    pub fn r#ReduceMax(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.ReduceMax", args) }
    /// Native Mimir.Layers.ReduceMean().
    pub fn r#ReduceMean(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.ReduceMean", args) }
    /// Native Mimir.Layers.ReduceMin().
    pub fn r#ReduceMin(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.ReduceMin", args) }
    /// Native Mimir.Layers.ReduceProd().
    pub fn r#ReduceProd(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.ReduceProd", args) }
    /// Native Mimir.Layers.ReduceSum().
    pub fn r#ReduceSum(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.ReduceSum", args) }
    /// Native Mimir.Layers.ReflectionPad2d().
    pub fn r#ReflectionPad2d(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.ReflectionPad2d", args) }
    /// Native Mimir.Layers.Reparameterize(...).
    pub fn r#Reparameterize(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.Reparameterize", args) }
    /// Native Mimir.Layers.ReplicationPad2d().
    pub fn r#ReplicationPad2d(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.ReplicationPad2d", args) }
    /// Native Mimir.Layers.Reshape().
    pub fn r#Reshape(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.Reshape", args) }
    /// Native Mimir.Layers.SelfAttention().
    pub fn r#SelfAttention(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.SelfAttention", args) }
    /// Native Mimir.Layers.SiLU().
    pub fn r#SiLU(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.SiLU", args) }
    /// Native Mimir.Layers.Sigmoid().
    pub fn r#Sigmoid(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.Sigmoid", args) }
    /// Native Mimir.Layers.Sign().
    pub fn r#Sign(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.Sign", args) }
    /// Native Mimir.Layers.Sin().
    pub fn r#Sin(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.Sin", args) }
    /// Native Mimir.Layers.Softmax().
    pub fn r#Softmax(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.Softmax", args) }
    /// Native Mimir.Layers.Softplus().
    pub fn r#Softplus(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.Softplus", args) }
    /// Native Mimir.Layers.Split().
    pub fn r#Split(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.Split", args) }
    /// Native Mimir.Layers.SqrtAbs().
    pub fn r#SqrtAbs(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.SqrtAbs", args) }
    /// Native Mimir.Layers.Square().
    pub fn r#Square(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.Square", args) }
    /// Native Mimir.Layers.SquaredReLU().
    pub fn r#SquaredReLU(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.SquaredReLU", args) }
    /// Native Mimir.Layers.Squeeze().
    pub fn r#Squeeze(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.Squeeze", args) }
    /// Native Mimir.Layers.Stack().
    pub fn r#Stack(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.Stack", args) }
    /// Native Mimir.Layers.Subtract().
    pub fn r#Subtract(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.Subtract", args) }
    /// Native Mimir.Layers.SwiGLU().
    pub fn r#SwiGLU(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.SwiGLU", args) }
    /// Native Mimir.Layers.Tanh().
    pub fn r#Tanh(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.Tanh", args) }
    /// Native Mimir.Layers.TokenMeanPool().
    pub fn r#TokenMeanPool(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.TokenMeanPool", args) }
    /// Native Mimir.Layers.Transpose().
    pub fn r#Transpose(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.Transpose", args) }
    /// Native Mimir.Layers.Unsqueeze().
    pub fn r#Unsqueeze(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.Unsqueeze", args) }
    /// Native Mimir.Layers.UpsampleBicubic().
    pub fn r#UpsampleBicubic(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.UpsampleBicubic", args) }
    /// Native Mimir.Layers.UpsampleBilinear().
    pub fn r#UpsampleBilinear(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.UpsampleBilinear", args) }
    /// Native Mimir.Layers.UpsampleNearest().
    pub fn r#UpsampleNearest(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.UpsampleNearest", args) }
    /// Native Mimir.Layers.View().
    pub fn r#View(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.View", args) }
    /// Native Mimir.Layers.ZeroPad2d().
    pub fn r#ZeroPad2d(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.ZeroPad2d", args) }
    /// Native Mimir.Layers.abs(...).
    pub fn r#abs(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.abs", args) }
    /// Native Mimir.Layers.activation(...).
    pub fn r#activation(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.activation", args) }
    /// Native Mimir.Layers.adaptiveavgpool2d(...).
    pub fn r#adaptiveavgpool2d(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.adaptiveavgpool2d", args) }
    /// Native Mimir.Layers.add(...).
    pub fn r#add(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.add", args) }
    /// Native Mimir.Layers.alphadropout(...).
    pub fn r#alphadropout(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.alphadropout", args) }
    /// Native Mimir.Layers.attention(...).
    pub fn r#attention(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.attention", args) }
    /// Native Mimir.Layers.available().
    pub fn r#available(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.available", args) }
    /// Native Mimir.Layers.avgpool1d(...).
    pub fn r#avgpool1d(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.avgpool1d", args) }
    /// Native Mimir.Layers.avgpool2d(...).
    pub fn r#avgpool2d(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.avgpool2d", args) }
    /// Native Mimir.Layers.batchmatmul(...).
    pub fn r#batchmatmul(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.batchmatmul", args) }
    /// Native Mimir.Layers.batchnorm(...).
    pub fn r#batchnorm(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.batchnorm", args) }
    /// Native Mimir.Layers.batchnorm1d(...).
    pub fn r#batchnorm1d(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.batchnorm1d", args) }
    /// Native Mimir.Layers.batchnorm2d(...).
    pub fn r#batchnorm2d(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.batchnorm2d", args) }
    /// Native Mimir.Layers.bilinear(...).
    pub fn r#bilinear(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.bilinear", args) }
    /// Native Mimir.Layers.by_type(layer_type).
    pub fn r#by_type(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.by_type", args) }
    /// Native Mimir.Layers.causaldepthwiseconv1d(...).
    pub fn r#causaldepthwiseconv1d(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.causaldepthwiseconv1d", args) }
    /// Native Mimir.Layers.chunk(...).
    pub fn r#chunk(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.chunk", args) }
    /// Native Mimir.Layers.concat(...).
    pub fn r#concat(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.concat", args) }
    /// Native Mimir.Layers.constant(...).
    pub fn r#constant(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.constant", args) }
    /// Native Mimir.Layers.conv1d(...).
    pub fn r#conv1d(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.conv1d", args) }
    /// Native Mimir.Layers.conv2d(...).
    pub fn r#conv2d(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.conv2d", args) }
    /// Native Mimir.Layers.conv3d(...).
    pub fn r#conv3d(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.conv3d", args) }
    /// Native Mimir.Layers.conv4d(...).
    pub fn r#conv4d(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.conv4d", args) }
    /// Native Mimir.Layers.convtranspose2d(...).
    pub fn r#convtranspose2d(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.convtranspose2d", args) }
    /// Native Mimir.Layers.cos(...).
    pub fn r#cos(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.cos", args) }
    /// Native Mimir.Layers.crossattention(...).
    pub fn r#crossattention(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.crossattention", args) }
    /// Native Mimir.Layers.depthwiseconv2d(...).
    pub fn r#depthwiseconv2d(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.depthwiseconv2d", args) }
    /// Native Mimir.Layers.divide(...).
    pub fn r#divide(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.divide", args) }
    /// Native Mimir.Layers.dropout(...).
    pub fn r#dropout(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.dropout", args) }
    /// Native Mimir.Layers.dropout2d(...).
    pub fn r#dropout2d(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.dropout2d", args) }
    /// Native Mimir.Layers.embedding(...).
    pub fn r#embedding(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.embedding", args) }
    /// Native Mimir.Layers.embeddingbag(...).
    pub fn r#embeddingbag(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.embeddingbag", args) }
    /// Native Mimir.Layers.exp(...).
    pub fn r#exp(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.exp", args) }
    /// Native Mimir.Layers.flatten(...).
    pub fn r#flatten(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.flatten", args) }
    /// Native Mimir.Layers.geglu(...).
    pub fn r#geglu(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.geglu", args) }
    /// Native Mimir.Layers.gelu(...).
    pub fn r#gelu(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.gelu", args) }
    /// Native Mimir.Layers.globalavgpool2d(...).
    pub fn r#globalavgpool2d(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.globalavgpool2d", args) }
    /// Native Mimir.Layers.groupnorm(...).
    pub fn r#groupnorm(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.groupnorm", args) }
    /// Native Mimir.Layers.gru(...).
    pub fn r#gru(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.gru", args) }
    /// Native Mimir.Layers.hardsigmoid(...).
    pub fn r#hardsigmoid(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.hardsigmoid", args) }
    /// Native Mimir.Layers.hardswish(...).
    pub fn r#hardswish(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.hardswish", args) }
    /// Native Mimir.Layers.identity(...).
    pub fn r#identity(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.identity", args) }
    /// Native Mimir.Layers.instancenorm2d(...).
    pub fn r#instancenorm2d(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.instancenorm2d", args) }
    /// Native Mimir.Layers.lambda(...).
    pub fn r#lambda(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.lambda", args) }
    /// Native Mimir.Layers.layernorm(...).
    pub fn r#layernorm(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.layernorm", args) }
    /// Native Mimir.Layers.leakyrelu(...).
    pub fn r#leakyrelu(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.leakyrelu", args) }
    /// Native Mimir.Layers.linear(...).
    pub fn r#linear(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.linear", args) }
    /// Native Mimir.Layers.logabs(...).
    pub fn r#logabs(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.logabs", args) }
    /// Native Mimir.Layers.logsoftmax(...).
    pub fn r#logsoftmax(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.logsoftmax", args) }
    /// Native Mimir.Layers.lowertriangular(...).
    pub fn r#lowertriangular(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.lowertriangular", args) }
    /// Native Mimir.Layers.lstm(...).
    pub fn r#lstm(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.lstm", args) }
    /// Native Mimir.Layers.matmul(...).
    pub fn r#matmul(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.matmul", args) }
    /// Native Mimir.Layers.maximum(...).
    pub fn r#maximum(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.maximum", args) }
    /// Native Mimir.Layers.maxpool1d(...).
    pub fn r#maxpool1d(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.maxpool1d", args) }
    /// Native Mimir.Layers.maxpool2d(...).
    pub fn r#maxpool2d(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.maxpool2d", args) }
    /// Native Mimir.Layers.minimum(...).
    pub fn r#minimum(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.minimum", args) }
    /// Native Mimir.Layers.mish(...).
    pub fn r#mish(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.mish", args) }
    /// Native Mimir.Layers.multiheadattention(...).
    pub fn r#multiheadattention(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.multiheadattention", args) }
    /// Native Mimir.Layers.multiply(...).
    pub fn r#multiply(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.multiply", args) }
    /// Native Mimir.Layers.nms(...).
    pub fn r#nms(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.nms", args) }
    /// Native Mimir.Layers.patchembed(...).
    pub fn r#patchembed(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.patchembed", args) }
    /// Native Mimir.Layers.permute(...).
    pub fn r#permute(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.permute", args) }
    /// Native Mimir.Layers.pixelshuffle(...).
    pub fn r#pixelshuffle(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.pixelshuffle", args) }
    /// Native Mimir.Layers.reciprocalnonan(...).
    pub fn r#reciprocalnonan(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.reciprocalnonan", args) }
    /// Native Mimir.Layers.reducemax(...).
    pub fn r#reducemax(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.reducemax", args) }
    /// Native Mimir.Layers.reducemean(...).
    pub fn r#reducemean(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.reducemean", args) }
    /// Native Mimir.Layers.reducemin(...).
    pub fn r#reducemin(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.reducemin", args) }
    /// Native Mimir.Layers.reduceprod(...).
    pub fn r#reduceprod(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.reduceprod", args) }
    /// Native Mimir.Layers.reducesum(...).
    pub fn r#reducesum(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.reducesum", args) }
    /// Native Mimir.Layers.reflectionpad2d(...).
    pub fn r#reflectionpad2d(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.reflectionpad2d", args) }
    /// Native Mimir.Layers.relu(...).
    pub fn r#relu(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.relu", args) }
    /// Native Mimir.Layers.reparameterize(...).
    pub fn r#reparameterize(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.reparameterize", args) }
    /// Native Mimir.Layers.replicationpad2d(...).
    pub fn r#replicationpad2d(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.replicationpad2d", args) }
    /// Native Mimir.Layers.reshape(...).
    pub fn r#reshape(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.reshape", args) }
    /// Native Mimir.Layers.rmsnorm(...).
    pub fn r#rmsnorm(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.rmsnorm", args) }
    /// Native Mimir.Layers.rnn(...).
    pub fn r#rnn(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.rnn", args) }
    /// Native Mimir.Layers.selfattention(...).
    pub fn r#selfattention(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.selfattention", args) }
    /// Native Mimir.Layers.sigmoid(...).
    pub fn r#sigmoid(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.sigmoid", args) }
    /// Native Mimir.Layers.sign(...).
    pub fn r#sign(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.sign", args) }
    /// Native Mimir.Layers.silu(...).
    pub fn r#silu(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.silu", args) }
    /// Native Mimir.Layers.sin(...).
    pub fn r#sin(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.sin", args) }
    /// Native Mimir.Layers.softmax(...).
    pub fn r#softmax(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.softmax", args) }
    /// Native Mimir.Layers.softplus(...).
    pub fn r#softplus(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.softplus", args) }
    /// Native Mimir.Layers.split(...).
    pub fn r#split(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.split", args) }
    /// Native Mimir.Layers.sqrtabs(...).
    pub fn r#sqrtabs(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.sqrtabs", args) }
    /// Native Mimir.Layers.square(...).
    pub fn r#square(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.square", args) }
    /// Native Mimir.Layers.squaredrelu(...).
    pub fn r#squaredrelu(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.squaredrelu", args) }
    /// Native Mimir.Layers.squeeze(...).
    pub fn r#squeeze(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.squeeze", args) }
    /// Native Mimir.Layers.stack(...).
    pub fn r#stack(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.stack", args) }
    /// Native Mimir.Layers.subtract(...).
    pub fn r#subtract(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.subtract", args) }
    /// Native Mimir.Layers.swiglu(...).
    pub fn r#swiglu(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.swiglu", args) }
    /// Native Mimir.Layers.tanh(...).
    pub fn r#tanh(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.tanh", args) }
    /// Native Mimir.Layers.tokenmeanpool(...).
    pub fn r#tokenmeanpool(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.tokenmeanpool", args) }
    /// Native Mimir.Layers.transpose(...).
    pub fn r#transpose(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.transpose", args) }
    /// Native Mimir.Layers.unsqueeze(...).
    pub fn r#unsqueeze(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.unsqueeze", args) }
    /// Native Mimir.Layers.upsamplebicubic(...).
    pub fn r#upsamplebicubic(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.upsamplebicubic", args) }
    /// Native Mimir.Layers.upsamplebilinear(...).
    pub fn r#upsamplebilinear(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.upsamplebilinear", args) }
    /// Native Mimir.Layers.upsamplenearest(...).
    pub fn r#upsamplenearest(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.upsamplenearest", args) }
    /// Native Mimir.Layers.view(...).
    pub fn r#view(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.view", args) }
    /// Native Mimir.Layers.zeropad2d(...).
    pub fn r#zeropad2d(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Layers.zeropad2d", args) }
}
#[allow(non_snake_case)] pub mod Memory {
    /// Native Mimir.Memory.clear().
    pub fn r#clear(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Memory.clear", args) }
    /// Native Mimir.Memory.config(cfg).
    pub fn r#config(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Memory.config", args) }
    /// Native Mimir.Memory.getStats().
    pub fn r#getStats(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Memory.getStats", args) }
    /// Native Mimir.Memory.getUsage().
    pub fn r#getUsage(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Memory.getUsage", args) }
    /// Native Mimir.Memory.get_stats().
    pub fn r#get_stats(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Memory.get_stats", args) }
    /// Native Mimir.Memory.get_usage().
    pub fn r#get_usage(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Memory.get_usage", args) }
    /// Native Mimir.Memory.printStats().
    pub fn r#printStats(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Memory.printStats", args) }
    /// Native Mimir.Memory.print_stats().
    pub fn r#print_stats(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Memory.print_stats", args) }
    /// Native Mimir.Memory.setLimit(limit_mb).
    pub fn r#setLimit(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Memory.setLimit", args) }
    /// Native Mimir.Memory.set_limit(limit_mb).
    pub fn r#set_limit(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Memory.set_limit", args) }
}
#[allow(non_snake_case)] pub mod MemoryGuard {
    /// Native Mimir.MemoryGuard.getCurrentUsage().
    pub fn r#getCurrentUsage(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("MemoryGuard.getCurrentUsage", args) }
    /// Native Mimir.MemoryGuard.getLimit().
    pub fn r#getLimit(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("MemoryGuard.getLimit", args) }
    /// Native Mimir.MemoryGuard.getPeakUsage().
    pub fn r#getPeakUsage(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("MemoryGuard.getPeakUsage", args) }
    /// Native Mimir.MemoryGuard.getStats().
    pub fn r#getStats(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("MemoryGuard.getStats", args) }
    /// Native Mimir.MemoryGuard.printStats().
    pub fn r#printStats(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("MemoryGuard.printStats", args) }
    /// Native Mimir.MemoryGuard.reset().
    pub fn r#reset(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("MemoryGuard.reset", args) }
    /// Native Mimir.MemoryGuard.setLimit(limit).
    pub fn r#setLimit(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("MemoryGuard.setLimit", args) }
}
#[allow(non_snake_case)] pub mod Model {
    /// Native Mimir.Model.allocate_params().
    pub fn r#allocate_params(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Model.allocate_params", args) }
    /// Native Mimir.Model.backward(loss_gradient).
    pub fn r#backward(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Model.backward", args) }
    /// Native Mimir.Model.build().
    pub fn r#build(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Model.build", args) }
    /// Native Mimir.Model.causal_lm_chat(message, memory_path, options).
    pub fn r#causal_lm_chat(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Model.causal_lm_chat", args) }
    /// Native Mimir.Model.causal_lm_generate(prompt, options).
    pub fn r#causal_lm_generate(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Model.causal_lm_generate", args) }
    /// Native Mimir.Model.causal_lm_memory_size(memory_path).
    pub fn r#causal_lm_memory_size(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Model.causal_lm_memory_size", args) }
    /// Native Mimir.Model.causal_lm_prepare_corpus(corpus_path, tokenizer_path, bpe_merges, seed).
    pub fn r#causal_lm_prepare_corpus(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Model.causal_lm_prepare_corpus", args) }
    /// Native Mimir.Model.causal_lm_reset_memory(memory_path).
    pub fn r#causal_lm_reset_memory(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Model.causal_lm_reset_memory", args) }
    /// Native Mimir.Model.causal_lm_train(steps, learning_rate, seed).
    pub fn r#causal_lm_train(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Model.causal_lm_train", args) }
    /// Native Mimir.Model.clear_layers().
    pub fn r#clear_layers(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Model.clear_layers", args) }
    /// Native Mimir.Model.conditioning_prompt(caption, annotation).
    pub fn r#conditioning_prompt(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Model.conditioning_prompt", args) }
    /// Native Mimir.Model.create(model_type, config).
    pub fn r#create(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Model.create", args) }
    /// Native Mimir.Model.create_empty(model_type, config).
    pub fn r#create_empty(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Model.create_empty", args) }
    /// Native Mimir.Model.create_from_config(cfg).
    pub fn r#create_from_config(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Model.create_from_config", args) }
    /// Native Mimir.Model.dtype(dtype).
    pub fn r#dtype(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Model.dtype", args) }
    /// Native Mimir.Model.forward(input, training).
    pub fn r#forward(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Model.forward", args) }
    /// Native Mimir.Model.get_config().
    pub fn r#get_config(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Model.get_config", args) }
    /// Native Mimir.Model.get_gradients().
    pub fn r#get_gradients(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Model.get_gradients", args) }
    /// Native Mimir.Model.get_layers().
    pub fn r#get_layers(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Model.get_layers", args) }
    /// Native Mimir.Model.hardware_caps().
    pub fn r#hardware_caps(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Model.hardware_caps", args) }
    /// Native Mimir.Model.infer(input).
    pub fn r#infer(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Model.infer", args) }
    /// Native Mimir.Model.init_weights(init, seed).
    pub fn r#init_weights(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Model.init_weights", args) }
    /// Native Mimir.Model.load(dir).
    pub fn r#load(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Model.load", args) }
    /// Native Mimir.Model.lumen_add_vae_calibration_image(image).
    pub fn r#lumen_add_vae_calibration_image(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Model.lumen_add_vae_calibration_image", args) }
    /// Native Mimir.Model.lumen_begin_vae_calibration().
    pub fn r#lumen_begin_vae_calibration(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Model.lumen_begin_vae_calibration", args) }
    /// Native Mimir.Model.lumen_finish_vae_calibration().
    pub fn r#lumen_finish_vae_calibration(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Model.lumen_finish_vae_calibration", args) }
    /// Native Mimir.Model.lumen_text2img(prompt, seed, steps, guidance).
    pub fn r#lumen_text2img(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Model.lumen_text2img", args) }
    /// Native Mimir.Model.lumen_train_step(image, prompt, seed, learning_rate, optimizer, monitor).
    pub fn r#lumen_train_step(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Model.lumen_train_step", args) }
    /// Native Mimir.Model.lumen_validate_step(image, prompt, seed, options).
    pub fn r#lumen_validate_step(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Model.lumen_validate_step", args) }
    /// Native Mimir.Model.optimizer_step(learning_rate, opt_type).
    pub fn r#optimizer_step(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Model.optimizer_step", args) }
    /// Native Mimir.Model.push_layer(name, layer_type, params_count, params).
    pub fn r#push_layer(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Model.push_layer", args) }
    /// Native Mimir.Model.runtime_ops().
    pub fn r#runtime_ops(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Model.runtime_ops", args) }
    /// Native Mimir.Model.save(dir).
    pub fn r#save(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Model.save", args) }
    /// Native Mimir.Model.set_autocast(dtype).
    pub fn r#set_autocast(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Model.set_autocast", args) }
    /// Native Mimir.Model.set_hardware(enable).
    pub fn r#set_hardware(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Model.set_hardware", args) }
    /// Native Mimir.Model.set_layer_io(layer_name, inputs, output).
    pub fn r#set_layer_io(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Model.set_layer_io", args) }
    /// Native Mimir.Model.total_params().
    pub fn r#total_params(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Model.total_params", args) }
    /// Native Mimir.Model.train(epochs, learning_rate).
    pub fn r#train(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Model.train", args) }
    /// Native Mimir.Model.vision_train_step(input, target, options).
    pub fn r#vision_train_step(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Model.vision_train_step", args) }
    /// Native Mimir.Model.zero_grads().
    pub fn r#zero_grads(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Model.zero_grads", args) }
}
#[allow(non_snake_case)] pub mod Serialization {
    /// Native Mimir.Serialization.detect_format(path).
    pub fn r#detect_format(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Serialization.detect_format", args) }
    /// Native Mimir.Serialization.load(path, format, options).
    pub fn r#load(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Serialization.load", args) }
    /// Native Mimir.Serialization.save(path, format, options).
    pub fn r#save(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Serialization.save", args) }
    /// Native Mimir.Serialization.save_enhanced_debug(path, options).
    pub fn r#save_enhanced_debug(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Serialization.save_enhanced_debug", args) }
}
#[allow(non_snake_case)] pub mod Tokenizer {
    /// Native Mimir.Tokenizer.add_token(token).
    pub fn r#add_token(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Tokenizer.add_token", args) }
    /// Native Mimir.Tokenizer.analyze_text(text).
    pub fn r#analyze_text(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Tokenizer.analyze_text", args) }
    /// Native Mimir.Tokenizer.batch_tokenize(texts).
    pub fn r#batch_tokenize(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Tokenizer.batch_tokenize", args) }
    /// Native Mimir.Tokenizer.bpe_merge_count().
    pub fn r#bpe_merge_count(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Tokenizer.bpe_merge_count", args) }
    /// Native Mimir.Tokenizer.create(max_vocab).
    pub fn r#create(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Tokenizer.create", args) }
    /// Native Mimir.Tokenizer.detokenize(tokens).
    pub fn r#detokenize(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Tokenizer.detokenize", args) }
    /// Native Mimir.Tokenizer.ensure_vocab_from_text(text).
    pub fn r#ensure_vocab_from_text(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Tokenizer.ensure_vocab_from_text", args) }
    /// Native Mimir.Tokenizer.extract_keywords(text, top_k).
    pub fn r#extract_keywords(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Tokenizer.extract_keywords", args) }
    /// Native Mimir.Tokenizer.get_frequencies().
    pub fn r#get_frequencies(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Tokenizer.get_frequencies", args) }
    /// Native Mimir.Tokenizer.get_max_vocab().
    pub fn r#get_max_vocab(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Tokenizer.get_max_vocab", args) }
    /// Native Mimir.Tokenizer.get_token_by_id(id).
    pub fn r#get_token_by_id(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Tokenizer.get_token_by_id", args) }
    /// Native Mimir.Tokenizer.learn_bpe(corpus, num_merges).
    pub fn r#learn_bpe(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Tokenizer.learn_bpe", args) }
    /// Native Mimir.Tokenizer.load(path).
    pub fn r#load(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Tokenizer.load", args) }
    /// Native Mimir.Tokenizer.mag_id().
    pub fn r#mag_id(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Tokenizer.mag_id", args) }
    /// Native Mimir.Tokenizer.mod_id().
    pub fn r#mod_id(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Tokenizer.mod_id", args) }
    /// Native Mimir.Tokenizer.pad_id().
    pub fn r#pad_id(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Tokenizer.pad_id", args) }
    /// Native Mimir.Tokenizer.pad_sequence(tokens, max_length, pad_id).
    pub fn r#pad_sequence(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Tokenizer.pad_sequence", args) }
    /// Native Mimir.Tokenizer.print_stats().
    pub fn r#print_stats(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Tokenizer.print_stats", args) }
    /// Native Mimir.Tokenizer.save(path).
    pub fn r#save(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Tokenizer.save", args) }
    /// Native Mimir.Tokenizer.seq_id().
    pub fn r#seq_id(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Tokenizer.seq_id", args) }
    /// Native Mimir.Tokenizer.set_max_length(max_length).
    pub fn r#set_max_length(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Tokenizer.set_max_length", args) }
    /// Native Mimir.Tokenizer.set_max_vocab(max_vocab).
    pub fn r#set_max_vocab(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Tokenizer.set_max_vocab", args) }
    /// Native Mimir.Tokenizer.tokenize(text).
    pub fn r#tokenize(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Tokenizer.tokenize", args) }
    /// Native Mimir.Tokenizer.tokenize_bpe(text).
    pub fn r#tokenize_bpe(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Tokenizer.tokenize_bpe", args) }
    /// Native Mimir.Tokenizer.tokenize_ensure(text).
    pub fn r#tokenize_ensure(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Tokenizer.tokenize_ensure", args) }
    /// Native Mimir.Tokenizer.unk_id().
    pub fn r#unk_id(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Tokenizer.unk_id", args) }
    /// Native Mimir.Tokenizer.vocab_size().
    pub fn r#vocab_size(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Tokenizer.vocab_size", args) }
}
#[allow(non_snake_case)] pub mod Viz {
    /// Native Mimir.Viz.add_image(pixels, width, height, channels, prompt).
    pub fn r#add_image(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Viz.add_image", args) }
    /// Native Mimir.Viz.add_loss_point(loss).
    pub fn r#add_loss_point(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Viz.add_loss_point", args) }
    /// Native Mimir.Viz.backend().
    pub fn r#backend(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Viz.backend", args) }
    /// Native Mimir.Viz.clear().
    pub fn r#clear(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Viz.clear", args) }
    /// Native Mimir.Viz.configure(scene).
    pub fn r#configure(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Viz.configure", args) }
    /// Native Mimir.Viz.create(config).
    pub fn r#create(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Viz.create", args) }
    /// Native Mimir.Viz.initialize().
    pub fn r#initialize(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Viz.initialize", args) }
    /// Native Mimir.Viz.is_open().
    pub fn r#is_open(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Viz.is_open", args) }
    /// Native Mimir.Viz.poll_events(timeout_ms).
    pub fn r#poll_events(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Viz.poll_events", args) }
    /// Native Mimir.Viz.process_events().
    pub fn r#process_events(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Viz.process_events", args) }
    /// Native Mimir.Viz.save_loss_history(path).
    pub fn r#save_loss_history(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Viz.save_loss_history", args) }
    /// Native Mimir.Viz.set_detection_threshold(threshold).
    pub fn r#set_detection_threshold(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Viz.set_detection_threshold", args) }
    /// Native Mimir.Viz.set_detections(detections, target).
    pub fn r#set_detections(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Viz.set_detections", args) }
    /// Native Mimir.Viz.set_enabled(enabled).
    pub fn r#set_enabled(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Viz.set_enabled", args) }
    /// Native Mimir.Viz.set_validation(state).
    pub fn r#set_validation(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Viz.set_validation", args) }
    /// Native Mimir.Viz.show_detections(enabled).
    pub fn r#show_detections(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Viz.show_detections", args) }
    /// Native Mimir.Viz.update().
    pub fn r#update(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Viz.update", args) }
    /// Native Mimir.Viz.update_metrics(metrics).
    pub fn r#update_metrics(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Viz.update_metrics", args) }
    /// Native Mimir.Viz.validation_enabled().
    pub fn r#validation_enabled(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("Viz.validation_enabled", args) }
}
#[allow(non_snake_case)] pub mod model {
    /// Native Mimir.model.allocate_params(...).
    pub fn r#allocate_params(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("model.allocate_params", args) }
    /// Native Mimir.model.backward(...).
    pub fn r#backward(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("model.backward", args) }
    /// Native Mimir.model.build(...).
    pub fn r#build(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("model.build", args) }
    /// Native Mimir.model.causal_lm_chat(...).
    pub fn r#causal_lm_chat(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("model.causal_lm_chat", args) }
    /// Native Mimir.model.causal_lm_generate(...).
    pub fn r#causal_lm_generate(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("model.causal_lm_generate", args) }
    /// Native Mimir.model.causal_lm_memory_size(...).
    pub fn r#causal_lm_memory_size(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("model.causal_lm_memory_size", args) }
    /// Native Mimir.model.causal_lm_prepare_corpus(...).
    pub fn r#causal_lm_prepare_corpus(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("model.causal_lm_prepare_corpus", args) }
    /// Native Mimir.model.causal_lm_reset_memory(...).
    pub fn r#causal_lm_reset_memory(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("model.causal_lm_reset_memory", args) }
    /// Native Mimir.model.causal_lm_train(...).
    pub fn r#causal_lm_train(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("model.causal_lm_train", args) }
    /// Native Mimir.model.clear_layers(...).
    pub fn r#clear_layers(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("model.clear_layers", args) }
    /// Native Mimir.model.conditioning_prompt(...).
    pub fn r#conditioning_prompt(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("model.conditioning_prompt", args) }
    /// Native Mimir.model.create(...).
    pub fn r#create(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("model.create", args) }
    /// Native Mimir.model.create_empty(...).
    pub fn r#create_empty(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("model.create_empty", args) }
    /// Native Mimir.model.create_from_config(...).
    pub fn r#create_from_config(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("model.create_from_config", args) }
    /// Native Mimir.model.dtype(dtype).
    pub fn r#dtype(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("model.dtype", args) }
    /// Native Mimir.model.forward(...).
    pub fn r#forward(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("model.forward", args) }
    /// Native Mimir.model.get_config(...).
    pub fn r#get_config(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("model.get_config", args) }
    /// Native Mimir.model.get_gradients(...).
    pub fn r#get_gradients(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("model.get_gradients", args) }
    /// Native Mimir.model.get_layers(...).
    pub fn r#get_layers(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("model.get_layers", args) }
    /// Native Mimir.model.hardware_caps(...).
    pub fn r#hardware_caps(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("model.hardware_caps", args) }
    /// Native Mimir.model.infer(...).
    pub fn r#infer(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("model.infer", args) }
    /// Native Mimir.model.init_weights(...).
    pub fn r#init_weights(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("model.init_weights", args) }
    /// Native Mimir.model.load(...).
    pub fn r#load(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("model.load", args) }
    /// Native Mimir.model.lumen_add_vae_calibration_image(...).
    pub fn r#lumen_add_vae_calibration_image(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("model.lumen_add_vae_calibration_image", args) }
    /// Native Mimir.model.lumen_begin_vae_calibration(...).
    pub fn r#lumen_begin_vae_calibration(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("model.lumen_begin_vae_calibration", args) }
    /// Native Mimir.model.lumen_finish_vae_calibration(...).
    pub fn r#lumen_finish_vae_calibration(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("model.lumen_finish_vae_calibration", args) }
    /// Native Mimir.model.lumen_text2img(...).
    pub fn r#lumen_text2img(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("model.lumen_text2img", args) }
    /// Native Mimir.model.lumen_train_step(...).
    pub fn r#lumen_train_step(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("model.lumen_train_step", args) }
    /// Native Mimir.model.lumen_validate_step(...).
    pub fn r#lumen_validate_step(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("model.lumen_validate_step", args) }
    /// Native Mimir.model.optimizer_step(...).
    pub fn r#optimizer_step(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("model.optimizer_step", args) }
    /// Native Mimir.model.push_layer(...).
    pub fn r#push_layer(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("model.push_layer", args) }
    /// Native Mimir.model.runtime_ops(...).
    pub fn r#runtime_ops(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("model.runtime_ops", args) }
    /// Native Mimir.model.save(...).
    pub fn r#save(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("model.save", args) }
    /// Native Mimir.model.set_autocast(...).
    pub fn r#set_autocast(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("model.set_autocast", args) }
    /// Native Mimir.model.set_hardware(...).
    pub fn r#set_hardware(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("model.set_hardware", args) }
    /// Native Mimir.model.set_layer_io(...).
    pub fn r#set_layer_io(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("model.set_layer_io", args) }
    /// Native Mimir.model.total_params(...).
    pub fn r#total_params(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("model.total_params", args) }
    /// Native Mimir.model.train(...).
    pub fn r#train(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("model.train", args) }
    /// Native Mimir.model.vision_train_step(...).
    pub fn r#vision_train_step(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("model.vision_train_step", args) }
    /// Native Mimir.model.zero_grads(...).
    pub fn r#zero_grads(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("model.zero_grads", args) }
}
pub use Architectures as architectures;
pub use Tokenizer as tokenizer;
pub use Dataset as dataset;
pub use Htop as htop;
pub use Viz as viz;
pub fn log(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("log", args) }
pub fn read_json(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("read_json", args) }
pub fn write_json(args: &[crate::Value]) -> crate::Result<crate::Value> { crate::call("write_json", args) }
#[allow(non_snake_case)] pub mod Mimir { pub use super::*; }
