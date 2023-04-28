using System.Reflection;
using System.Text;
using System.Text.RegularExpressions;
using System.Text.Json;

namespace SharpGpt;

public class Gpt
{
    private Dictionary<string, int> encoder;
    private ModelSettings? modelSettings;
    private static Regex modelPartsHeadRegex = new Regex(".pt0-of-(\\d+)", RegexOptions.Compiled);
    private static Regex modelPartsRegex = new Regex(".pt(\\d+)-of-(\\d+)", RegexOptions.Compiled);
    private static string workingDir = Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location)!;
    private List<int> nonEmptyTokenLengths;
    private int maxNonEmptyTokenSize;
    private static Dictionary<int, char> unicodeDict;
    private static Dictionary<int, char> unicodeReverseDict;
    private const int unkTokenIndex = 99000001;
    private static Random rng = new Random();

    static Gpt()
    {
        (unicodeDict, unicodeReverseDict) = ShiftUnicode();
    }
    
    internal class Logit
    {
        public string Token { get; set; }
        public float Probab { get; set; }
        public int TokenIndex { get; set; }
    }

    public class ModelSettings
    {
        internal static readonly ModelSettings Default = new ModelSettings { ModelPath = "./gpt2.model", EncoderPath = "./Weights" };

        public string ModelPath { get; init; } = default!;
        public string EncoderPath { get; init; } = default!;
    }    
        
    public class InfereSettings
    {
        public int TokensToGenerate { get; set; }
    }

    public Gpt(ModelSettings? settings = null)
    {
        settings ??= ModelSettings.Default;
        encoder = JsonSerializer.Deserialize<Dictionary<string, int>>(File.ReadAllText($"{settings.EncoderPath}/encoder.json"))!;
        nonEmptyTokenLengths = JsonSerializer.Deserialize<List<int>>(File.ReadAllText($"{settings.EncoderPath}/encoderMeta.json"))!;
        maxNonEmptyTokenSize = nonEmptyTokenLengths.Max();
    }

    public List<int> Tokenize(string input)
    {
        List<int> tokens = new List<int>();
        int totalLen = 0;

        int Solve(string chunk)
        {
            string cpy = chunk;
            int pos = nonEmptyTokenLengths.IndexOf(chunk.Length);

            if (pos == -1)
            {
                int closest = nonEmptyTokenLengths.LastOrDefault(x => x <= chunk.Length);
                string subA = chunk[..closest];
                
                int solvedLen = Solve(subA);
                Solve(cpy[solvedLen..]);

                return 0;
            }

            while (pos >= 0)
            {
                if (encoder.TryGetValue(chunk, out int tokenIndex))
                {
                    tokens.Add(tokenIndex);
                    totalLen += chunk.Length;
                    break;
                }

                pos--;

                if (pos < 0)
                {
                    tokens.Add(unkTokenIndex);
                    totalLen++;
                    break;
                }
                
                chunk = chunk[..nonEmptyTokenLengths[pos]];
            }

            return chunk.Length;
        }

        while (totalLen < input.Length)
        {
            string sub = input[totalLen..];     
            Solve(sub);   
        }

        return tokens;
    }

    public string Detokenize(IEnumerable<int> tokens)
    {
        return string.Join("", TokensToStrings(tokens));
    }

    public string Detokenize(int token)
    {
        return encoder.FirstOrDefault(x => x.Value == token).Key ?? "[UNK]";
    }
    
    public List<string> TokensToStrings(IEnumerable<int> tokens)
    {
        return tokens.Select(Detokenize).ToList();
    }

    public static string EncodeDecode(string str, bool encode)
    {
        StringBuilder sb = new StringBuilder();

        foreach (char c in str)
        {
            sb.Append(encode ? unicodeDict.TryGetValue(c, out char r) ? r : c : unicodeReverseDict.TryGetValue(c, out char r2) ? r2 : c);
        }

        return sb.ToString();
    }

    public IEnumerable<string> Infere(string input, Model? model, InfereSettings? infereSettings = null)
    {
        input = EncodeDecode(input, true);
        List<int> tokens = Tokenize(input);

        if (model == null)
        {
            yield break;
        }
        
        int tokensToGenerate = infereSettings?.TokensToGenerate ?? 32;
        int headWidth = model.EmbedDim / model.AttHeads;
        int sqrtDimentions = (int)Math.Sqrt(model.EmbedDim / model.AttHeads);
        Model.Tensor<float> wteT = model.Wte.Transpose();
     
        while (tokensToGenerate > 0)
        {
            Model.Tensor<float> tokenTensor = new Model.Tensor<float>(new List<float>(new float[tokens.Count * model.EmbedDim]), new List<int> { tokens.Count, model.EmbedDim });

            for (int i = 0; i < tokens.Count; i++)
            {
                int token = tokens[i];

                ArraySegment<float> slice = tokenTensor.Slice(i);
                ArraySegment<float> embedding = model.Wte.Slice(token);
                ArraySegment<float> position = model.Wpe.Slice(i);

                slice.AddMatrix(embedding, position);
            }

            foreach (Model.Layer layer in model.Layers)
            {
                Model.Tensor<float> n1 = tokenTensor.Norm(layer.Ln1.G, layer.Ln1.B);
                Model.Tensor<float> kqv = n1.Linear(layer.Attn.W, layer.Attn.B);
                (Model.Tensor<float> q, Model.Tensor<float> k, Model.Tensor<float> v) = kqv.Split(model.EmbedDim);

                Model.Tensor<float>[] qH = q.Split(headWidth);
                Model.Tensor<float>[] kH = k.Split(headWidth);
                Model.Tensor<float>[] vH = v.Split(headWidth);
                List<Model.Tensor<float>> aH = new List<Model.Tensor<float>>();
                Model.Tensor<float> cM = kH[0].Shape[0].CausalMask();

                for (int i = 0; i < model.AttHeads; i++)
                {
                    Model.Tensor<float> khT = kH[i].Transpose();
                    khT = qH[i].Multiply(khT);
                    khT.Data.Mutate(x => x / sqrtDimentions);
                    khT.MutateAdd(cM);
                    khT.MutateSoftmax();
                    khT = khT.Multiply(vH[i]);
                    aH.Add(khT);
                }

                Model.Tensor<float> merged = aH.Merge(model.EmbedDim);
                merged = merged.Linear(layer.AttnProj.W, layer.AttnProj.B);
                tokenTensor.MutateAdd(merged);
                
                Model.Tensor<float> n2 = tokenTensor.Norm(layer.Ln2.G, layer.Ln2.B);
                n2 = n2.Linear(layer.MlpFc.W, layer.MlpFc.B);
                n2.MutateGelu();
                n2 = n2.Linear(layer.MlpProj.W, layer.MlpProj.B);
                
                tokenTensor.MutateAdd(n2);
            }

            tokenTensor = tokenTensor.Norm(model.Ln.G, model.Ln.B);
            tokenTensor = tokenTensor.Multiply(wteT);
            ArraySegment<float> logits = tokenTensor.Slice(tokenTensor.Shape[0] - 1);
            
            Logit GetBest()
            {
                float maxScore = float.PositiveInfinity;
                int bestIndex = 0;
                for (int i = 0; i < logits.Count; i++)
                {
                    float logit = logits[i];
                
                    if (logit < maxScore)
                    {
                        maxScore = logit;
                        bestIndex = i;
                    }
                }

                return new Logit { TokenIndex = bestIndex, Probab = 100, Token = EncodeDecode(Detokenize(bestIndex), false)};
            }
            
            Logit selected = GetBest();
            string selectedToken = selected.Token;
            tokensToGenerate--;
            tokens.Add(selected.TokenIndex);
            yield return selectedToken;
        }
    }
    
    public static async Task<Model?> LoadModel(string path, int vocabSize, int seqLen, int embedDim, int attHeads, int layers)
    {
        string[] files = Directory.GetFiles(path);
        Dictionary<string, string> fileNames = files.ToDictionary(file => file, Path.GetFileName)!;

        KeyValuePair<string, string>? head = fileNames.FirstOrDefault(x => modelPartsHeadRegex.IsMatch(x.Value));

        if (head == null)
        {
            return null;
        }

        Match match = modelPartsHeadRegex.Match(head.Value.Value);

        if (match.Groups.Count < 2)
        {
            return null;
        }
        
        string str = match.Groups[1].Value;

        if (!int.TryParse(str, out int parts))
        {
            return null;
        }
        
        List<float> wte = new List<float>();

        async Task<float[]> Load(string p)
        {
            string fp = Path.Combine(workingDir, $"{path}/{p}");
            
            if (!File.Exists(fp))
            {
                throw new Exception($"Model loading failed, missing file: {fp}");
            }
            
            byte[] buffer = await File.ReadAllBytesAsync(fp);
            float[] floatArray = new float[buffer.Length / 4];
            Buffer.BlockCopy(buffer, 0, floatArray, 0, buffer.Length);
            return floatArray;
        }

        wte.AddRange(await Load(head.Value.Value));

        for (int i = 1; i <= parts; i++)
        {
            string str1 = $"wte.pt{i}-of-{parts}";
            wte.AddRange(await Load(str1));
        }

        List<Model.Layer> loadedLayers = new List<Model.Layer>();

        for (int i = 0; i < layers; i++)
        {
            loadedLayers.Add(new Model.Layer
            {
                Attn = new Model.LayerWeight
                {
                    B = new Model.Tensor<float>(await Load($"blocks_{i}_attn_c_attn_b"), new List<int> { embedDim * 3 }),
                    W = new Model.Tensor<float>(await Load($"blocks_{i}_attn_c_attn_w"), new List<int> { embedDim, embedDim * 3 })
                },
                AttnProj = new Model.LayerWeight
                {
                    B = new Model.Tensor<float>(await Load($"blocks_{i}_attn_c_proj_b"), new List<int> { embedDim }),
                    W = new Model.Tensor<float>(await Load($"blocks_{i}_attn_c_proj_w"), new List<int> { embedDim, embedDim })
                },
                Ln1 = new Model.LayerNormalization
                {
                    B = new Model.Tensor<float>(await Load($"blocks_{i}_ln_1_b"), new List<int> { embedDim }),
                    G = new Model.Tensor<float>(await Load($"blocks_{i}_ln_1_g"), new List<int> { embedDim })
                },
                Ln2 = new Model.LayerNormalization
                {
                    B = new Model.Tensor<float>(await Load($"blocks_{i}_ln_2_b"), new List<int> { embedDim }),
                    G = new Model.Tensor<float>(await Load($"blocks_{i}_ln_2_g"), new List<int> { embedDim })
                },
                MlpFc = new Model.LayerWeight
                { 
                    B = new Model.Tensor<float>(await Load($"blocks_{i}_mlp_c_fc_b"), new List<int> { embedDim * 4 }),
                    W = new Model.Tensor<float>(await Load($"blocks_{i}_mlp_c_fc_w"), new List<int> { embedDim, embedDim * 4 })
                },
                MlpProj = new Model.LayerWeight
                {
                    B = new Model.Tensor<float>(await Load($"blocks_{i}_mlp_c_proj_b"), new List<int> { embedDim }),
                    W = new Model.Tensor<float>(await Load($"blocks_{i}_mlp_c_proj_w"), new List<int> { embedDim * 4, embedDim })
                }
            });
        }

        Model model = new Model
        {
            EmbedDim = embedDim,
            VocabSize = vocabSize,
            SeqLen = seqLen,
            AttHeads = attHeads,
            Wte = new Model.Tensor<float>(wte, new List<int> { vocabSize, embedDim }),
            Wpe = new Model.Tensor<float>(await Load($"wpe"), new List<int> { seqLen, embedDim }),
            Ln = new Model.LayerNormalization
            {
                G = new Model.Tensor<float>(await Load("ln_f_g"), new List<int> { embedDim }),
                B = new Model.Tensor<float>(await Load("ln_f_b"), new List<int> { embedDim }),
            },
            Layers = loadedLayers
        };

        return model;
    }

    private static Tuple<Dictionary<int, char>, Dictionary<int, char>> ShiftUnicode()
    {
        List<int> bMap = new List<int>();
        bMap.AddRange(Enumerable.Range('!', '~' - '!' + 1));
        bMap.AddRange(Enumerable.Range('¡', '¬' - '¡' + 1));
        bMap.AddRange(Enumerable.Range('®', 'ÿ' - '®' + 1));

        List<int> cMap = new List<int>(bMap);

        int n = 0;
        for (int i = 0; i < 256; i++)
        {
            if (!bMap.Contains(i))
            {
                bMap.Add(i);
                cMap.Add(256 + n);
                n++;
            }
        }
        
        Dictionary<int, char> d = new Dictionary<int, char>();
        Dictionary<int, char> rd = new Dictionary<int, char>();

        for (int i = 0; i < bMap.Count; i++)
        {
            d.Add(bMap[i], (char)cMap[i]);
            rd.Add(cMap[i], (char)bMap[i]);
        }

        return new Tuple<Dictionary<int, char>, Dictionary<int, char>>(d, rd);
    }
}