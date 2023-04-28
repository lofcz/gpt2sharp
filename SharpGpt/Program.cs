namespace SharpGpt;

class Program
{
    public static async Task Main()
    {
        Console.Title = "GPT2#";
        Console.WriteLine("Loading model..");
        
        Gpt gpt = new Gpt();
        Model? model = await Gpt.LoadModel("Weights", 50257, 1024, 768, 12, 12);

        Console.Clear();
        Console.Write("> ");
        string? input = Console.ReadLine();
        Console.Clear();
        Console.Write($"> {input}");

        Console.ForegroundColor = ConsoleColor.Cyan;
        foreach (string token in gpt.Infere(input ?? "", model))
        {
            Console.Write(token);
        }
        Console.ForegroundColor = ConsoleColor.White;

        Console.ReadKey();
    }
}