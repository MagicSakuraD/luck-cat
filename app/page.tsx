"use client";
import { useEffect, useState } from "react";
import * as tf from "@tensorflow/tfjs";

import { historyData } from "./data/historyData";
import ModeToggle from "@/components/mode-toggle";

// 添加 Prophet 预测结果
const prophetPredictions = {
  blue: { yhat: 7.817406, yhat_lower: 1.813098, yhat_upper: 16.058283 },
  red: [
    { yhat: 4.369632, yhat_lower: 0.021479, yhat_upper: 9.143139 },
    { yhat: 8.563668, yhat_lower: 2.007665, yhat_upper: 14.364846 },
    { yhat: 14.306099, yhat_lower: 7.23877, yhat_upper: 20.787135 },
    { yhat: 19.705636, yhat_lower: 13.069152, yhat_upper: 26.610567 },
    { yhat: 23.826644, yhat_lower: 17.426201, yhat_upper: 30.416162 },
    { yhat: 29.145027, yhat_lower: 24.383593, yhat_upper: 33.148594 },
  ],
};

export default function Home() {
  const [model, setModel] = useState<any>(null);
  const [prediction, setPrediction] = useState<number[] | null>(null);
  const [isSaving, setIsSaving] = useState<boolean>(false);
  const [saveStatus, setSaveStatus] = useState<string>("");
  const trainNumber = 32;

  useEffect(() => {
    const initializeModel = async () => {
      if (typeof window !== "undefined" && window.ml5) {
        await tf.ready();

        // 创建 ml5.js 模型
        const nn = window.ml5.neuralNetwork({
          task: "regression",
          debug: true,
          learningRate: 0.0001, // 稍微提高学习率
          layers: [
            {
              type: "dense",
              units: 256,
              activation: "relu",
            },
            {
              type: "dense",
              units: 128,
              activation: "relu",
            },
            {
              type: "dense",
              units: 64,
              activation: "relu",
            },
            {
              type: "dense",
              units: 32,
              activation: "relu",
            },
            {
              type: "dense",
              units: 16,
              activation: "relu",
            },
            {
              type: "dense",
              units: 7,
              activation: "linear",
            },
          ],
        });

        setModel(nn);
      }
    };

    initializeModel();
  }, []);

  const calculateWeights = (
    historyData: { reds: number[]; blue: number }[]
  ) => {
    const redBallCounts = Array(33).fill(0);
    const blueBallCounts = Array(16).fill(0);

    for (let i = 0; i < trainNumber; i++) {
      historyData[i].reds.forEach((num: number) => {
        redBallCounts[num - 1] += 1;
      });
      blueBallCounts[historyData[i].blue - 1] += 1;
    }

    // 权重
    const redBallWeights = redBallCounts.map((count) => count / trainNumber);
    const blueBallWeights = blueBallCounts.map((count) => count / trainNumber);
    return { redBallWeights, blueBallWeights };
  };

  // 调整输入数据以反映权重
  const adjustWithWeights = (data: any[], weights: number[]) => {
    return data.map(
      (value: number, index: string | number) =>
        value * weights[index as number]
      // value * 1
    );
  };

  useEffect(() => {
    const trainAndPredict = async () => {
      if (model) {
        const preprocessData = (data: number[]) => {
          const redBalls = data.slice(0, 6).sort((a, b) => a - b);
          const blueBall = data[6];
          return [...redBalls, blueBall];
        };

        const normalize = (value: number, max: number) => value / max;
        const denormalize = (value: number, max: number) =>
          Math.round(value * max);

        // 计算权重
        const { redBallWeights, blueBallWeights } =
          calculateWeights(historyData);

        // 添加数据
        for (let i = historyData.length - 1; i >= trainNumber; i--) {
          let inputs = [];
          for (let j = i; j > i - trainNumber; j--) {
            let processedData = preprocessData([
              ...historyData[j].reds,
              historyData[j].blue,
            ]);

            let adjustedData = adjustWithWeights(
              processedData.slice(0, 6).map((x) => normalize(x, 33)),
              redBallWeights
            );
            let adjustedBlueBall =
              normalize(processedData[6], 16) *
              blueBallWeights[processedData[6] - 1];

            inputs.push(...adjustedData, adjustedBlueBall);
          }
          let outputs = preprocessData([
            ...historyData[i - trainNumber].reds,
            historyData[i - trainNumber].blue,
          ]);
          let adjustedOutputs = adjustWithWeights(
            outputs.slice(0, 6).map((x) => normalize(x, 33)),
            redBallWeights
          );
          let adjustedBlueOutput =
            normalize(outputs[6], 16) * blueBallWeights[outputs[6] - 1];

          model.addData(inputs, [...adjustedOutputs, adjustedBlueOutput]);
        }

        model.normalizeData();

        const trainingOptions = {
          epochs: 140,
          batchSize: 32,
          shuffle: true,
          validationSplit: 0.2,
        };

        const finishedTraining = () => {
          console.log("Model trained!");
          makePrediction(preprocessData, normalize, denormalize);
        };

        await model.train(trainingOptions, finishedTraining);
      }
    };

    const makePrediction = (
      preprocessData: { (data: number[]): number[]; (arg0: number[]): any },
      normalize: {
        (value: number, max: number): number;
        (arg0: number, arg1: number): any;
      },
      denormalize: {
        (value: number, max: number): number;
        (arg0: any, arg1: number): number;
      }
    ) => {
      // 进行预测
      const lastEntries = historyData.slice(0, trainNumber);
      const inputData = lastEntries.flatMap((entry) => {
        const processedData = preprocessData([...entry.reds, entry.blue]);
        return [
          ...processedData.slice(0, 6).map((x) => normalize(x, 33)),
          normalize(processedData[6], 16),
        ];
      });

      model.predict(inputData, (results: any, err: any) => {
        if (err) {
          console.error(err, "something went wrong");
        } else {
          if (Array.isArray(results) && results.length === 7) {
            const adjustedPrediction = results.map((r, index) => {
              let value;
              if (index < 6) {
                // 红球
                value = Math.max(1, Math.min(33, denormalize(r.value, 33)));
                console.log("Red ball value:", value);
                const prophetPred = prophetPredictions.red[index];
                // 确保预测值在 Prophet 模型的置信区间内
                value = Math.max(
                  Math.min(value, prophetPred.yhat_upper),
                  prophetPred.yhat_lower
                );
              } else {
                // 蓝球
                value = Math.max(1, Math.min(16, denormalize(r.value, 16)));
                console.log("Blue ball value:", value);
                // 确保预测值在 Prophet 模型的置信区间内
                value = Math.max(
                  Math.min(value, prophetPredictions.blue.yhat_upper),
                  prophetPredictions.blue.yhat_lower
                );
              }
              return Math.round(value);
            });
            const redBalls = adjustedPrediction.slice(0, 6);
            setPrediction([...redBalls, adjustedPrediction[6]]);
            console.log(
              "Adjusted prediction:",
              redBalls,
              adjustedPrediction[6]
            );
          } else {
            console.error("Prediction did not return 7 numbers:", results);
          }
        }
      });
    };

    trainAndPredict();
  }, [model]);

  const handleSaveModel = async () => {
    if (model) {
      setIsSaving(true);
      setSaveStatus("Saving model...");
      try {
        await model.save();
        setSaveStatus("Model saved successfully!");
      } catch (error) {
        console.error("Error saving model:", error);
        setSaveStatus("Error saving model. Please try again.");
      } finally {
        setIsSaving(false);
      }
    } else {
      setSaveStatus("No model to save. Please train the model first.");
    }
  };

  return (
    <main className="flex min-h-screen flex-col items-center justify-between p-24">
      <div>
        <h1>Prediction</h1>
        {prediction && (
          <p>
            Red: {prediction.slice(0, 6).join(", ")} <br />
            Blue: {prediction[6]}
          </p>
        )}
        <button
          onClick={handleSaveModel}
          disabled={isSaving || !model}
          className="mt-4 px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-270 disabled:bg-gray-400"
        >
          {isSaving ? "Saving..." : "Save Model"}
        </button>
        {saveStatus && <p className="mt-2">{saveStatus}</p>}
      </div>
    </main>
  );
}
