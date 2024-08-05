"use client";
import { useEffect, useState } from "react";
import * as tf from "@tensorflow/tfjs";

import { historyData } from "./data/historyData";

export default function Home() {
  const [model, setModel] = useState<any>(null);
  const [prediction, setPrediction] = useState<number[] | null>(null);
  const [isSaving, setIsSaving] = useState<boolean>(false);
  const [saveStatus, setSaveStatus] = useState<string>("");
  const trainNumber = 24;

  useEffect(() => {
    const initializeModel = async () => {
      if (typeof window !== "undefined" && window.ml5) {
        await tf.ready();

        // 创建 ml5.js 模型
        const nn = window.ml5.neuralNetwork({
          task: "regression",
          debug: true,
          learningRate: 0.001, // 可以尝试较低的学习率
          layers: [
            { type: "dense", units: 512, activation: "relu" }, // 增加神经元数量
            {
              type: "dense",
              units: 256, // 第一隐藏层神经元数量
              activation: "relu",
            },
            {
              type: "dense",
              units: 128, // 第二隐藏层神经元数量
              activation: "relu",
            },
            {
              type: "dense",
              units: 64, // 第三隐藏层神经元数量
              activation: "relu",
            },
            {
              type: "dense",
              units: 32, // 第四隐藏层神经元数量
              activation: "relu",
            },
            {
              type: "dense",
              units: 16, // 第五隐藏层神经元数量
              activation: "relu",
            },
            {
              type: "dense",
              units: 7, // 输出层神经元数量
              activation: "linear",
            },
          ],
        });

        setModel(nn);
      }
    };

    initializeModel();
  }, []);

  // 计算数字出现频率并分配权重
  const calculateWeights = (
    historyData: { reds: number[]; blue: number }[]
  ) => {
    const redBallCounts = Array(33).fill(0);
    const blueBallCounts = Array(16).fill(0);

    for (let i = 0; i < trainNumber; i++) {
      historyData[i].reds.forEach((num: number) => {
        redBallCounts[num - 1] += 1 + trainNumber;
      });
      blueBallCounts[historyData[i].blue - 1] += 1;
    }

    // 归一化权重
    const redBallWeights = redBallCounts.map(
      (count) => count / (2 * trainNumber)
    );
    const blueBallWeights = blueBallCounts.map((count) => count / trainNumber);

    return { redBallWeights, blueBallWeights };
  };

  // 调整输入数据以反映权重
  const adjustWithWeights = (data: any[], weights: number[]) => {
    return data.map(
      (value: number, index: string | number) =>
        value * weights[index as number]
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
          epochs: 100,
          callbacks: {
            onEpochEnd: async (epoch: number, logs: any) => {
              console.log(`Epoch ${epoch}: loss = ${logs.loss}`);
              if (logs.loss < 0.04) {
                await model.save();
                console.log(
                  `Model saved at epoch ${epoch} with loss ${logs.loss}`
                );
                model.stopTraining = true;
              }
            },
          },
        };

        await model.train(trainingOptions);
        console.log("Model trained!");

        // 进行预测
        const last10Entries = historyData.slice(0, trainNumber);
        const inputData = last10Entries.flatMap((entry) => {
          const processedData = preprocessData([...entry.reds, entry.blue]);
          return [
            ...processedData.slice(0, 6).map((x) => normalize(x, 33)),
            normalize(processedData[6], 16),
          ];
        });

        model.predict(inputData, (err: any, results: any) => {
          if (err) {
            console.error(err);
          } else {
            console.log("Raw prediction results:", results);

            if (Array.isArray(results) && results.length === 7) {
              const adjustedPrediction = results.map(
                (r: any, index: number) => {
                  if (index < 6) {
                    return Math.max(1, Math.min(33, denormalize(r.value, 33)));
                  } else {
                    return Math.max(1, Math.min(16, denormalize(r.value, 16)));
                  }
                }
              );

              // 确保红球升序且不重复
              const redBalls = Array.from(
                new Set(adjustedPrediction.slice(0, 6))
              ).sort((a, b) => a - b);
              while (redBalls.length < 6) {
                let newNum = Math.floor(Math.random() * 33) + 1;
                if (!redBalls.includes(newNum)) {
                  redBalls.push(newNum);
                }
              }

              setPrediction([...redBalls, adjustedPrediction[6]]);
            } else {
              console.error("Prediction did not return 7 numbers:", results);
            }
          }
        });
      }
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
            Predicted numbers: Red balls: {prediction.slice(0, 6).join(", ")}{" "}
            <br />
            Blue ball: {prediction[6]}
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
