using MathNet.Numerics.Statistics;
using Microsoft.ML;
using Microsoft.ML.Data;
using ScottPlot;
using System;
using System.Globalization;
using System.Linq;

namespace ConsoleApp3
{
    class Program
    {
        static void Main(string[] args)
        {
            // 1. Создание контекста ML.NET
            MLContext mlContext = new MLContext();

            // 2. Загрузка данных
            IDataView data = mlContext.Data.LoadFromTextFile<GapminderData>("dataset.csv", separatorChar: ',', hasHeader: true);

            // 3. Первичный анализ и вывод информации
            Console.WriteLine($"Количество строк: {data.GetColumn<string>(nameof(GapminderData.Country)).ToList().Count}");
            Console.WriteLine($"Количество столбцов: {data.Schema.Count}");


            var countryStats = data.GetColumn<string>(nameof(GapminderData.Country)).Distinct().ToList().Count;
            Console.WriteLine($"Количество уникальных стран: {countryStats}");


            var continentStats = data.GetColumn<string>(nameof(GapminderData.Continent)).Distinct().ToList().Count;
            Console.WriteLine($"Количество уникальных континентов: {continentStats}");

            var yearAsFloats = data.GetColumn<float>(nameof(GapminderData.Year)).ToArray(); ;

            var validYears = yearAsFloats.Where(year => !float.IsNaN(year) && year > 0).ToArray();

            double yearMean = validYears.Average();
            Console.WriteLine($"Средний год: {yearMean}");

            double[] yearAsDoubles = yearAsFloats
                .Select(year =>
                {
                    if (!float.IsNaN(year) && year > 0)
                    {
                        return (double)year;
                    }

                    return yearMean;
                }).ToArray();

            Console.WriteLine($"Стандартное отклонение года: {yearAsDoubles.StandardDeviation()}");


            var popStats = data.GetColumn<float>(nameof(GapminderData.Pop)).ToArray();
            Console.WriteLine($"Средняя численность населения: {popStats.Mean()}");
            Console.WriteLine($"Стандартное отклонение численности населения: {popStats.StandardDeviation()}");


            var gdpPercapStats = data.GetColumn<float>(nameof(GapminderData.GdpPercap)).ToArray();
            Console.WriteLine($"Средний ВВП на душу населения: {gdpPercapStats.Mean()}");
            Console.WriteLine($"Стандартное отклонение ВВП на душу населения: {gdpPercapStats.StandardDeviation()}");

            // 4. Преобразование категориальных признаков
            var pipeline = mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "CountryEncoded", inputColumnName: nameof(GapminderData.Country))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "ContinentEncoded", inputColumnName: nameof(GapminderData.Continent)))
                .Append(mlContext.Transforms.Concatenate("Features", "CountryEncoded", "ContinentEncoded", nameof(GapminderData.Year), nameof(GapminderData.Pop), nameof(GapminderData.GdpPercap))); // Объединение признаков

            IDataView transformedData = pipeline.Fit(data).Transform(data);


            // 5. Разделение на обучающую и тестовую выборки 80% 20% соответственно
            DataOperationsCatalog.TrainTestData splitData = mlContext.Data.TrainTestSplit(transformedData, testFraction: 0.2);
            IDataView trainData = splitData.TrainSet;
            IDataView testData = splitData.TestSet;

            // 6. Создание и обучение модели
            var trainingPipeline = mlContext.Transforms.Concatenate("Features", "CountryEncoded", "ContinentEncoded", nameof(GapminderData.Year), nameof(GapminderData.Pop), nameof(GapminderData.GdpPercap))
                .Append(mlContext.Regression.Trainers.Sdca(labelColumnName: nameof(GapminderData.LifeExp), featureColumnName: "Features"));

            ITransformer model = trainingPipeline.Fit(trainData);

            // 7. Оценка качества модели
            IDataView predictions = model.Transform(testData);
            var metrics = mlContext.Regression.Evaluate(predictions, labelColumnName: nameof(GapminderData.LifeExp));

            Console.WriteLine($"RMSE: {metrics.RootMeanSquaredError}");
            Console.WriteLine($"MAE: {metrics.MeanAbsoluteError}");
            Console.WriteLine($"R^2: {metrics.RSquared}");

            double[] lifeExp = data.GetColumn<float>(nameof(GapminderData.LifeExp)).Select(x => (double)x).ToArray();
            double[] gdpPercap = data.GetColumn<float>(nameof(GapminderData.GdpPercap)).Select(x => (double)x).ToArray();
            double[] pop = data.GetColumn<float>(nameof(GapminderData.Pop)).Select(x => (double)x).ToArray();
            double[] year = yearAsDoubles;


            // Гистограммы
            var plt1 = new ScottPlot.Plot(600, 400);
            plt1.AddSignal(lifeExp, sampleRate: 1);
            plt1.Title("Распределение продолжительности жизни");
            plt1.XLabel("Частота");
            plt1.YLabel("Продолжительность жизни");
            plt1.SaveFig("life_exp_dist.png");

            var plt2 = new ScottPlot.Plot(600, 400);
            plt2.AddSignal(gdpPercap, sampleRate: 1);
            plt2.Title("Распределение ВВП на душу населения");
            plt2.XLabel("Частота");
            plt2.YLabel("ВВП на душу населения");
            plt2.SaveFig("gdp_per_capita_dist.png");

            var plt3 = new ScottPlot.Plot(600, 400);
            plt3.AddSignal(pop, sampleRate: 1);
            plt3.Title("Распределение численности населения");
            plt3.XLabel("Частота");
            plt3.YLabel("Численность населения");
            plt3.SaveFig("pop_dist.png");


            var plt4 = new ScottPlot.Plot(600, 400);

            // Подсчет количества каждого года (для гистограммы)
            var yearCounts = year.GroupBy(y => y).Select(g => new { Year = g.Key, Count = g.Count() });


            // Добавление гистограммы
            var bar = plt4.AddBar(values: yearCounts.Select(x => (double)x.Count).ToArray(), positions: yearCounts.Select(x => x.Year).ToArray());

            bar.BarWidth = 0.8; // ширина столбиков


            // Подписи
            plt4.XLabel("Год");
            plt4.YLabel("Количество");
            plt4.Title("Распределение по годам");
            plt4.SaveFig("year_dist.png");

            // scatterplot
            var plt5 = new ScottPlot.Plot(600, 400);
            plt5.AddScatter(gdpPercap, lifeExp);
            plt5.XLabel("ВВП на душу населения");
            plt5.YLabel("Продолжительность жизни");
            plt5.Title("Корреляция: ВВП на душу населения vs продолжительность жизни");
            plt5.SaveFig("gdp_vs_life_exp.png");


            // Тепловая карта
            string[] featureColumns = transformedData.Schema.Select(x => x.Name).ToArray();
            double[,] corrMatrix = CorrelationMatrix(mlContext, transformedData, featureColumns);

            var plt6 = new ScottPlot.Plot(600, 400);
            var hm = plt6.AddHeatmap(corrMatrix);

            plt6.Title("Тепловая карта корреляционной матрицы");
            plt6.SaveFig("correlation_heatmap.png");

        }

        public static double[,] CorrelationMatrix(MLContext mlContext, IDataView data, string[] featureColumns)
        {
            int n = featureColumns.Length;
            double[,] matrix = new double[n, n];

            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    // Получаем данные столбцов. Обрабатываем float[] после OneHotEncoding
                    double[] col1 = GetColumnAsDoubleArray(data, featureColumns[i]);
                    double[] col2 = GetColumnAsDoubleArray(data, featureColumns[j]);

                    if (col1 != null && col2 != null)
                    {
                        matrix[i, j] = Correlation.Pearson(col1, col2);
                    }
                    else
                    {
                         // Обработка случая, если один из столбцов не числовой
                        matrix[i, j] = 0; // Или другое значение по умолчанию
                    }
                }
            }

            return matrix;
        }




        // Вспомогательная функция для получения данных столбца как double[]
        private static double[] GetColumnAsDoubleArray(IDataView data, string columnName)
        {
            var columnType = data.Schema[columnName].Type;

            if (columnType.RawType == typeof(float))
            {
                return data.GetColumn<float>(columnName).Select(x => (double)x).ToArray();
            }
            else if (columnType.Equals(typeof(ReadOnlyMemory<float>)))
            {
                return data.GetColumn<ReadOnlyMemory<float>>(columnName)
                                .SelectMany(x => x.ToArray())
                                .Select(x => (double)x)
                                .ToArray();
            }
            else if (columnType.RawType.IsGenericType && columnType.RawType.GetGenericTypeDefinition() == typeof(VBuffer<float>))
            {
                return data.GetColumn<VBuffer<float>>(columnName)
                           .SelectMany(x => x.DenseValues())
                           .Select(x => (double)x)
                           .ToArray();

            }
            else
            {
                return null;
            }
        }
    }
}