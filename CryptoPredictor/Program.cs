using System;
using System.Threading.Tasks;
using System.Collections.Generic;
using System.Globalization;
using System.Threading;
using Microsoft.ML;
using Microsoft.ML.Data;
using Newtonsoft.Json;
using MySql.Data.MySqlClient;
using System.Net.Http;
using MySqlConnector;

namespace CryptoPredictor
{
    class Program
    {
        static async Task Main(string[] args)
        {
            int M = 10; // Delay between session starts in seconds
            int numberOfSessions = 300; // Number of sessions
            Console.Title = $"PentaPredictions | Tasks: {numberOfSessions}";
            double initialBalance = 1000.0; // Initial balance for each session

            string connectionString = "Server=localhost;Port=3306;Database=gizmo_db;Uid=DBADM;Pwd=rootpass;";

            using (var conn = new MySql.Data.MySqlClient.MySqlConnection(connectionString))
            {
                conn.Open();
                var createTableCmd = new MySql.Data.MySqlClient.MySqlCommand(@"
                    CREATE TABLE IF NOT EXISTS session_logs (
                        session_id INT AUTO_INCREMENT PRIMARY KEY,
                        session_number INT,
                        start_time DATETIME,
                        action VARCHAR(10),
                        bet_percentage FLOAT,
                        balance FLOAT,
                        prediction_correct BOOLEAN
                    );
                ", conn);
                createTableCmd.ExecuteNonQuery();
                conn.Close();
            }

            List<Task> sessionTasks = new List<Task>();

            for (int i = 1; i <= numberOfSessions; i++)
            {
                int sessionNumber = i;
                Task sessionTask = Task.Run(() => RunSession(sessionNumber, initialBalance, connectionString));
                sessionTasks.Add(sessionTask);

                await Task.Delay(TimeSpan.FromSeconds(M));
            }

            await Task.WhenAll(sessionTasks);
        }

        static async Task RunSession(int sessionNumber, double initialBalance, string connectionString)
        {
            double balance = initialBalance;
            int N = 2; 
            double previousPrice = 0.0;
            var startTime = DateTime.UtcNow;

            var mlContext = new MLContext();
            ITransformer model = null;
            IDataView trainingDataView = null;
            List<CryptoData> data = new List<CryptoData>();

            while (true)
            {
                var newData = await GetDataFromAPI();
                if (newData.Count == 0)
                {
                    Console.WriteLine($"[Session#{sessionNumber}] Failed to get data. Retrying in N minutes.");
                    await Task.Delay(TimeSpan.FromMinutes(N));
                    continue;
                }

                data.AddRange(newData);
                previousPrice = newData[newData.Count - 1].Close;

                trainingDataView = mlContext.Data.LoadFromEnumerable(data);
                model = TrainModel(mlContext, trainingDataView);

                var prediction = Predict(mlContext, model, newData[newData.Count - 1]);

                float betPercentage = 0.03f + prediction.Probability * (0.10f - 0.03f);
                betPercentage = Math.Max(0.03f, Math.Min(betPercentage, 0.10f));

                Console.WriteLine($"[Session#{sessionNumber}] Prediction: {(prediction.PredictedLabel ? "Up" : "Down")}, Bet: {betPercentage * 100:F2}%");

                await Task.Delay(TimeSpan.FromMinutes(N));

                double actualPercentageChange = await GetActualPriceChange(previousPrice);

                bool predictionCorrect = (prediction.PredictedLabel && actualPercentageChange > 0) || (!prediction.PredictedLabel && actualPercentageChange < 0);

                balance = UpdateBalance(balance, betPercentage, prediction.PredictedLabel, actualPercentageChange);

                Console.WriteLine($"[Session#{sessionNumber}] New Balance: {balance:F2}");

                LogResultToDatabase(connectionString, sessionNumber, startTime, prediction.PredictedLabel ? "Up" : "Down", betPercentage, balance, predictionCorrect);

                if (!predictionCorrect)
                {
                    model = TrainModel(mlContext, trainingDataView);
                }

            }
        }

        static async Task<List<CryptoData>> GetDataFromAPI()
        {
            try
            {
                var client = new HttpClient();
                var response = await client.GetAsync("https://api.binance.com/api/v3/klines?symbol=ETHBTC&interval=1m&limit=100");
                var content = await response.Content.ReadAsStringAsync();

                var klineData = JsonConvert.DeserializeObject<List<List<object>>>(content);

                var dataList = new List<CryptoData>();

                foreach (var kline in klineData)
                {
                    var data = new CryptoData
                    {
                        OpenTime = Convert.ToInt64(kline[0]),
                        Open = float.Parse(kline[1].ToString(), CultureInfo.InvariantCulture),
                        High = float.Parse(kline[2].ToString(), CultureInfo.InvariantCulture),
                        Low = float.Parse(kline[3].ToString(), CultureInfo.InvariantCulture),
                        Close = float.Parse(kline[4].ToString(), CultureInfo.InvariantCulture),
                        Volume = float.Parse(kline[5].ToString(), CultureInfo.InvariantCulture),
                        CloseTime = Convert.ToInt64(kline[6]),
                        QuoteAssetVolume = float.Parse(kline[7].ToString(), CultureInfo.InvariantCulture),
                        NumberOfTrades = float.Parse(kline[8].ToString(), CultureInfo.InvariantCulture),
                        TakerBuyBaseAssetVolume = float.Parse(kline[9].ToString(), CultureInfo.InvariantCulture),
                        TakerBuyQuoteAssetVolume = float.Parse(kline[10].ToString(), CultureInfo.InvariantCulture),
                        Ignore = float.Parse(kline[11].ToString(), CultureInfo.InvariantCulture),
                        Label = float.Parse(kline[4].ToString(), CultureInfo.InvariantCulture) > float.Parse(kline[1].ToString(), CultureInfo.InvariantCulture)
                    };

                    dataList.Add(data);
                }

                return dataList;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error getting data: {ex.Message}");
                return new List<CryptoData>();
            }
        }

        static ITransformer TrainModel(MLContext mlContext, IDataView dataView)
        {
            var pipeline = mlContext.Transforms.Concatenate("Features", new string[] { "Open", "High", "Low", "Close", "Volume", "NumberOfTrades" })
                .Append(mlContext.Transforms.NormalizeMeanVariance("Features"))
                .Append(mlContext.BinaryClassification.Trainers.FastTree());

            var model = pipeline.Fit(dataView);

            return model;
        }

        static CryptoPrediction Predict(MLContext mlContext, ITransformer model, CryptoData input)
        {
            var predEngine = mlContext.Model.CreatePredictionEngine<CryptoData, CryptoPrediction>(model);

            var prediction = predEngine.Predict(input);

            return prediction;
        }

        static async Task<double> GetActualPriceChange(double previousPrice)
        {
            try
            {
                var client = new HttpClient();
                var response = await client.GetAsync("https://api.binance.com/api/v3/ticker/price?symbol=ETHBTC");
                var content = await response.Content.ReadAsStringAsync();

                var currentPriceData = JsonConvert.DeserializeObject<Dictionary<string, string>>(content);
                var currentPrice = double.Parse(currentPriceData["price"], CultureInfo.InvariantCulture);

                double percentageChange = (currentPrice - previousPrice) / previousPrice;

                return percentageChange;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error getting current price: {ex.Message}");
                return 0.0;
            }
        }

        static double UpdateBalance(double balance, float betPercentage, bool predictedUp, double actualPercentageChange)
        {
            double betAmount = balance * betPercentage;

            double profitOrLoss = betAmount * actualPercentageChange * (predictedUp ? 1 : -1) * 20;

            balance += profitOrLoss;

            return balance;
        }

        static void LogResultToDatabase(string connectionString, int sessionNumber, DateTime startTime, string action, float betPercentage, double balance, bool predictionCorrect)
        {
            using (var conn = new MySql.Data.MySqlClient.MySqlConnection(connectionString))
            {
                conn.Open();
                var cmd = new MySql.Data.MySqlClient.MySqlCommand(@"
                    INSERT INTO session_logs (session_number, start_time, action, bet_percentage, balance, prediction_correct)
                    VALUES (@session_number, @start_time, @action, @bet_percentage, @balance, @prediction_correct);
                ", conn);

                cmd.Parameters.AddWithValue("@session_number", sessionNumber);
                cmd.Parameters.AddWithValue("@start_time", startTime);
                cmd.Parameters.AddWithValue("@action", action);
                cmd.Parameters.AddWithValue("@bet_percentage", betPercentage);
                cmd.Parameters.AddWithValue("@balance", balance);
                cmd.Parameters.AddWithValue("@prediction_correct", predictionCorrect);

                cmd.ExecuteNonQuery();
                conn.Close();
            }
        }
    }

    public class CryptoData
    {
        [LoadColumn(0)]
        public long OpenTime;

        [LoadColumn(1)]
        public float Open;

        [LoadColumn(2)]
        public float High;

        [LoadColumn(3)]
        public float Low;

        [LoadColumn(4)]
        public float Close;

        [LoadColumn(5)]
        public float Volume;

        [LoadColumn(6)]
        public long CloseTime;

        [LoadColumn(7)]
        public float QuoteAssetVolume;

        [LoadColumn(8)]
        public float NumberOfTrades;

        [LoadColumn(9)]
        public float TakerBuyBaseAssetVolume;

        [LoadColumn(10)]
        public float TakerBuyQuoteAssetVolume;

        [LoadColumn(11)]
        public float Ignore;

        [ColumnName("Label")]
        public bool Label;
    }

    public class CryptoPrediction
    {
        [ColumnName("PredictedLabel")]
        public bool PredictedLabel;

        public float Probability;

        public float Score;
    }
}
