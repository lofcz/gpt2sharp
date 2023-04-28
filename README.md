# GPT2#

![logoD](https://user-images.githubusercontent.com/10260230/235049057-7fef9c7d-4f93-4974-9b2a-1263808b3b46.png)  
GPT2# is a sub-1,000-lines, zero dependencies GPT2 implementation written in C# 11, capable of loading the original weights released by OpenAI.

## ⚙️ Build
To get started with GPT2#, make sure `dotnet --version` returns 7.0+. If not, grab your copy of the SDK [here](https://dotnet.microsoft.com/en-us/download/visual-studio-sdks).  
Then, run the following commands:
- `git clone https://github.com/lofcz/gpt2sharp & cd gpt2sharp`
- `dotnet run --project SharpGpt`

## 🔮 About GPT2#
GPT2# is a minimal model inference implementation intended for ML hackers. Have you ever wondered how LLMs work behind the scenes?  
This repository can help you understand the process. The code is written in a clean, easily readable manner. The entire forward pass consists of just about 100 lines of code.

## ✖️ The Math Behind GPT2#
GPT2# has just a handful of functions defined, all of which can be found in `MathExt.cs`. For more information on specific functions, you can follow the links attached.  
These functions were defined by hand without using any third-party libraries on purpose, as understanding them is an important step to understanding the inference process.
- Matrices: [+](https://en.wikipedia.org/wiki/Matrix_addition), [*](https://en.wikipedia.org/wiki/Matrix_multiplication), [causal mask](https://medium.com/@jinoo/a-simple-example-of-attention-masking-in-transformer-decoder-a6c66757bc7d), [norm](https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html), [transpose](https://en.wikipedia.org/wiki/Transpose), [split](https://en.wikipedia.org/wiki/Matrix_splitting), [merge](https://www.geeksforgeeks.org/combining-matrices-in-r/), [linear](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html)
- Activation functions: [softmax](https://en.wikipedia.org/wiki/Softmax_function), [gelu](https://medium.com/@shauryagoel/gelu-gaussian-error-linear-unit-4ec59fb2e47c)

## 📸 Inference example
![gpt2](https://user-images.githubusercontent.com/10260230/235043657-328c2f78-d4c8-49b1-986e-fbb412a2dc99.png)  
_Text in cyan is inferred_

## ℹ️ Limitations
- Currently, only greedy decoding is implemented (no temp/top_k). To implement these, you can replace the implementation of the GetBest function in Gpt.cs. For guidance, consult the section "Sampling" [here](https://jaykmody.com/blog/gpt-from-scratch/).
- The math used in GPT2# is not accelerated by using a GPU/TPU or SIMD vectorization. While a `Parallel.For()` is used in matrix multiplication, the math is written in a simple, easy-to-understand way rather than being focused on performance. Replacing the naive matrix multiplication with the [Strassen algorithm](https://en.wikipedia.org/wiki/Strassen_algorithm) can provide a significant speedup.
- The used model is pretty small and dated, don't expect miracles. The first few tokens inferred are generally ok, but due to the lack of [repetition penalty](https://docs.ai21.com/docs/repetition-penalties) and greedy sampling the model stucks in a token loop pretty fast.

## Literature & Related projects
- https://jaykmody.com/blog/gpt-from-scratch - a great read - uses Python & NumPy/JAX, paired with GPT# should be easier to follow for .NET developers
- https://github.com/newhouseb/potatogpt - plugged their weights splitting script, thanks!
- https://developers.google.com/machine-learning/glossary - consult if looking for a specific term
