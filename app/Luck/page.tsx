"use client";
import { useEffect, useState } from "react";
import * as tf from "@tensorflow/tfjs";
import { Card, CardContent, CardFooter } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";

const prophetPredictions = {
  blue: { yhat: 11, yhat_lower: 2, yhat_upper: 14 },
  red: [
    { yhat: 7, yhat_lower: 1, yhat_upper: 8 },
    { yhat: 13, yhat_lower: 4, yhat_upper: 15 },
    { yhat: 15, yhat_lower: 7, yhat_upper: 26 },
    { yhat: 21, yhat_lower: 14, yhat_upper: 27 },
    { yhat: 31, yhat_lower: 19, yhat_upper: 31 },
    { yhat: 33, yhat_lower: 24, yhat_upper: 33 },
  ],
};

const historicalWeight = 0.7;
const prophetWeight = 0.3;
const blueBallWeightFactor = 33 / 16;

const normalize = (x: number, mean: number, std: number) =>
  (x - mean) / (std || 1);

const preprocessData = (
  data: { reds: number[]; blue: number },
  redMean: number,
  redStd: number,
  blueMean: number,
  blueStd: number
) => {
  const redBalls = data.reds.sort((a, b) => a - b);
  const blueBall = data.blue;

  const normalizedRedBalls = redBalls.map((ball) =>
    normalize(ball, redMean, redStd)
  );
  const normalizedBlueBall = normalize(blueBall, blueMean, blueStd);

  const prophetRedDiffs = prophetPredictions.red.map((pred, index) =>
    index === 0 ? pred.yhat : pred.yhat - prophetPredictions.red[index - 1].yhat
  );
  const prophetRedFeatures = prophetRedDiffs.map((diff) =>
    normalize(diff, redMean, redStd)
  );
  const prophetBlueFeature = normalize(
    prophetPredictions.blue.yhat,
    blueMean,
    blueStd
  );

  const weightedRedBalls = normalizedRedBalls.map(
    (val) => val * historicalWeight
  );
  const weightedBlueBall =
    normalizedBlueBall * historicalWeight * blueBallWeightFactor;

  const weightedProphetRedBalls = prophetRedFeatures.map(
    (val) => val * prophetWeight
  );
  const weightedProphetBlueBall =
    prophetBlueFeature * prophetWeight * blueBallWeightFactor;

  return [
    ...weightedRedBalls,
    weightedBlueBall,
    ...weightedProphetRedBalls,
    weightedProphetBlueBall,
  ];
};

export default function Home() {
  const [model, setModel] = useState<any>(null);
  const [prediction, setPrediction] = useState<number[] | null>(null);
  const [userInput, setUserInput] = useState<string>("");

  useEffect(() => {
    const initializeModel = async () => {
      if (typeof window !== "undefined" && window.ml5) {
        await tf.ready();

        const nn = window.ml5.neuralNetwork({
          task: "regression",
          debug: true,
        });

        const modelInfo = {
          model: "/model/model.json",
          metadata: "/model/model_meta.json",
          weights: "/model/model.weights.bin",
        };

        await nn.load(modelInfo);
        setModel(nn);
      }
    };

    initializeModel();
  }, []);

  const handleUserInput = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setUserInput(e.target.value);
  };

  const makePrediction = async () => {
    if (!model) return;

    try {
      const inputArray = JSON.parse(userInput);

      if (!Array.isArray(inputArray) || inputArray.length === 0) {
        alert("请输入有效的数组对象");
        return;
      }

      const allRedBalls = inputArray.flatMap((entry) => entry.reds);

      const allBlueBalls = inputArray.map((entry) => entry.blue);
      const redMean =
        allRedBalls.reduce((a, b) => a + b, 0) / allRedBalls.length;
      const redStd = Math.sqrt(
        allRedBalls.reduce((a, b) => a + Math.pow(b - redMean, 2), 0) /
          allRedBalls.length
      );
      const blueMean =
        allBlueBalls.reduce((a, b) => a + b, 0) / allBlueBalls.length;
      const blueStd = Math.sqrt(
        allBlueBalls.reduce((a, b) => a + Math.pow(b - blueMean, 2), 0) /
          allBlueBalls.length
      );

      const processedInput = inputArray.map((entry) =>
        preprocessData(entry, redMean, redStd, blueMean, blueStd)
      );

      const denormalize = (x: number, mean: number, std: number) =>
        x * (std || 1) + mean;

      model.predict(processedInput, (results: any, err: any) => {
        if (err) {
          console.error(err, "something went wrong");
        } else {
          if (Array.isArray(results) && results.length === 7) {
            const adjustedPrediction = results.map((r, index) => {
              let value;
              if (index < 6) {
                value = Math.round(denormalize(r.value, redMean, redStd));
                const prophetPred = prophetPredictions.red[index];
                value = Math.max(
                  Math.min(value, prophetPred.yhat_upper),
                  prophetPred.yhat_lower
                );
              } else {
                value = Math.round(denormalize(r.value, blueMean, blueStd));
                value = Math.max(
                  Math.min(value, prophetPredictions.blue.yhat_upper),
                  prophetPredictions.blue.yhat_lower
                );
              }
              return Math.max(1, Math.min(index < 6 ? 33 : 16, value));
            });
            setPrediction(adjustedPrediction);
          }
        }
      });
    } catch (error) {
      alert("输入格式错误，请输入有效的JSON数组对象");
      console.error("Error parsing input:", error);
    }
  };

  return (
    <Card className="flex justify-center items-center h-screen flex-col">
      <CardContent>
        <Textarea
          value={userInput}
          onChange={handleUserInput}
          placeholder='请输入历史数据，以JSON数组对象格式 (例如: [{"issue": "24119", "reds": [2, 9, 26, 27, 31, 32], "blue": 14}, {...}])'
          className="w-full p-2 mb-4 border rounded"
        />
        {prediction && (
          <div className="h-3/5">
            <p>Result: {prediction.join(", ")}</p>
          </div>
        )}
      </CardContent>

      <CardFooter>
        <Button onClick={makePrediction} disabled={!userInput}>
          prophet
        </Button>
      </CardFooter>
    </Card>
  );
}
