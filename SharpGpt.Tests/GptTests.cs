namespace SharpGpt.Tests;

public class GptTests
{
    [SetUp]
    public void Setup()
    {
    }

    [Test]
    [TestCase("Responsive range slider library written in typescript and using web component technologies.")]
    [TestCase("we are the")]
    [TestCase("Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.")]
    [TestCase("")]
    [TestCase("                  test")]
    [TestCase("         test         ")]
    public void Tokenize(string input)
    {
        Gpt gpt = new Gpt();
        string encoded = Gpt.EncodeDecode(input, true);
        
        List<int> tokens = gpt.Tokenize(encoded);
        string check = gpt.Detokenize(tokens);
        
        StringAssert.AreEqualIgnoringCase(encoded, check);
    }

    [TestCase("🔥🔥🔥")]
    [TestCase("huňatý kůň prováděl žluťounkou žábu přes česnekové pole")]
    public void TokenizeInvalid(string input)
    {
        Gpt gpt = new Gpt();
        string encoded = Gpt.EncodeDecode(input, true);
        
        List<int> tokens = gpt.Tokenize(encoded);
        string check = gpt.Detokenize(tokens);
        
        StringAssert.AreNotEqualIgnoringCase(encoded, check);
    }
}