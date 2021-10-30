using System;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Algorithms;
using Microsoft.ML.Probabilistic.Distributions;
// VS2017 (Framework 4.7) Infer.NET 0.3.1810.501

namespace InferStrengths
{
    class InferStrengthsProgram
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Begin Infer.NET demo ");

            // ===== Set up teams and win-loss data =====================

            string[] teamNames = new string[] { "Angels", "Bruins",
        "Comets", "Demons", "Eagles", "Flyers" };
            int N = teamNames.Length;

            int[] winTeamIDs = new int[] { 0, 2, 1, 0, 1, 3, 0, 2, 4 };
            int[] loseTeamIDs = new int[] { 1, 3, 2, 4, 3, 5, 5, 4, 5 };

            Console.WriteLine("Data: \n");
            for (int i = 0; i < winTeamIDs.Length; ++i)
            {
                Console.WriteLine("game: " + i + "   winning team: " +
                  teamNames[winTeamIDs[i]] + "   losing team: " +
                  teamNames[loseTeamIDs[i]]);
            }

            // ===== Define a probabilistic model =======================

            Range teamIDsRange = new Range(N).Named("teamsIDRange");
            Range gameIDsRange =
              new Range(winTeamIDs.Length).Named("gameIDsRange");

            double mean = 2000.0;
            double sd = 200.0;
            double vrnc = sd * sd;

            Console.WriteLine("\nDefining Gaussian model mean = " +
              mean.ToString("F1") + " and sd = " + sd.ToString("F1"));
            VariableArray<double> strengths =
              Variable.Array<double>(teamIDsRange).Named("strengths");

            strengths[teamIDsRange] =
              Variable.GaussianFromMeanAndVariance(mean,
              vrnc).ForEach(teamIDsRange);

            VariableArray<int> winners =
              Variable.Array<int>(gameIDsRange).Named("winners");
            VariableArray<int> losers =
              Variable.Array<int>(gameIDsRange).Named("losers");

            winners.ObservedValue = winTeamIDs;
            losers.ObservedValue = loseTeamIDs;

            using (Variable.ForEach(gameIDsRange))
            {
                var ws = strengths[winners[gameIDsRange]];
                var ls = strengths[losers[gameIDsRange]];
                Variable<double> winnerPerf =
              Variable.GaussianFromMeanAndVariance(ws, 400).Named("winPerf");
                Variable<double> loserPerf =
              Variable.GaussianFromMeanAndVariance(ls, 400).
                Named("losePerf");

                Variable.ConstrainTrue(winnerPerf > loserPerf);
            }

            // ===== Infer team strengths using win-loss data ===========

            Console.WriteLine("\nInferring strengths from win-loss data");
            var iengine = new InferenceEngine();
            iengine.Algorithm = new ExpectationPropagation();
            iengine.NumberOfIterations = 40;
            iengine.ShowFactorGraph = true;  // needs Graphviz install

            Gaussian[] inferredStrengths = iengine.Infer<Gaussian[]>(strengths);
            Console.WriteLine("Inference complete. Inferred strengths: ");

            // ===== Show results =======================================

            for (int i = 0; i < N; ++i)
            {
                double strength = inferredStrengths[i].GetMean();
                Console.WriteLine(teamNames[i] + ": " +
                  strength.ToString("F1"));
            }

            Console.WriteLine("\nEnd demo ");
            Console.ReadLine();
        } // Main
    }
} // ns
