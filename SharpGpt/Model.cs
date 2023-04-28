namespace SharpGpt;

public class Model
{
    public class Tensor<T>
    {
        public T[] Data { get; set; }
        public int[] Shape { get; set; }

        public Tensor(IEnumerable<T> data, IEnumerable<int> shape)
        {
            Data = data.ToArray();
            Shape = shape.ToArray();
        }

        public ArraySegment<T> Slice(int part)
        {
            int index = Shape.Length > 1 ? Shape[1] : 0;
            int startIndex = part * index;

            return new ArraySegment<T>(Data, startIndex, index);
        }
        
        public ArraySegment<T> Slice(int start, int count)
        {
            return new ArraySegment<T>(Data, start, count);
        }

        public Tensor<T> Duplicate()
        {
            return new Tensor<T>((T[])Data.Clone(), (int[])Shape.Clone());
        }
    }

    public class LayerNormalization
    {
        /// <summary>
        /// Gain
        /// </summary>
        public Tensor<float> G { get; set; }
        
        /// <summary>
        /// Bias
        /// </summary>
        public Tensor<float> B { get; set; }
    }

    public class LayerWeight
    {
        /// <summary>
        /// Weight
        /// </summary>
        public Tensor<float> W { get; set; }
        
        /// <summary>
        /// Bias
        /// </summary>
        public Tensor<float> B { get; set; }
    }

    public class Layer
    {
        /// <summary>
        /// Attention
        /// </summary>
        public LayerWeight Attn { get; set; }
        
        /// <summary>
        /// Attention projection
        /// </summary>
        public LayerWeight AttnProj { get; set; }
        
        /// <summary>
        /// Layer normalization 1
        /// </summary>
        public LayerNormalization Ln1 { get; set; }
        
        /// <summary>
        /// Layer normalization 2
        /// </summary>
        public LayerNormalization Ln2 { get; set; }
        
        /// <summary>
        /// Multi layer perceptron, full connected
        /// </summary>
        public LayerWeight MlpFc { get; set; }
        
        /// <summary>
        /// Multi layer perceptron, projection
        /// </summary>
        public LayerWeight MlpProj { get; set; }
    }
    
    /// <summary>
    /// Vocabulary size
    /// </summary>
    public int VocabSize { get; set; }
    
    /// <summary>
    /// Sequence length
    /// </summary>
    public int SeqLen { get; set; }
    
    /// <summary>
    /// Embedding dimensions
    /// </summary>
    public int EmbedDim { get; set; }
    
    /// <summary>
    /// Attention heads
    /// </summary>
    public int AttHeads { get; set; }

    /// <summary>
    /// Word token embedding
    /// </summary>
    public Tensor<float> Wte { get; set; }
    
    /// <summary>
    /// Word position embedding
    /// </summary>
    public Tensor<float> Wpe { get; set; }
    
    /// <summary>
    /// For feedforward network
    /// </summary>
    public LayerNormalization Ln { get; set; }
    
    /// <summary>
    /// Layers
    /// </summary>
    public List<Layer> Layers { get; set; }
}