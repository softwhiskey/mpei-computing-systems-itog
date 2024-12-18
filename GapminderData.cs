using Microsoft.ML.Data;

namespace ConsoleApp3
{
    public class GapminderData
    {
        [LoadColumn(0)] public string Country;
        [LoadColumn(1)] public string Continent;
        [LoadColumn(2)] public float Year;
        [LoadColumn(3)] public float LifeExp;
        [LoadColumn(4)] public float Pop;
        [LoadColumn(5)] public float GdpPercap;
    }
}
