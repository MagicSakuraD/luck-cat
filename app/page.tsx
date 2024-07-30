"use client";
import { useEffect, useState } from "react";
import * as tf from "@tensorflow/tfjs";

import { historyData } from "./data/historyData";

export default function Home() {
  const [model, setModel] = useState<any>(null);
  const [prediction, setPrediction] = useState<number[] | null>(null);
  const [isSaving, setIsSaving] = useState<boolean>(false);
  const [saveStatus, setSaveStatus] = useState<string>("");

  useEffect(() => {
    const initializeModel = async () => {
      if (typeof window !== "undefined" && window.ml5) {
        // 确保 TensorFlow.js 已经初始化
        await tf.ready();

        // 创建 ml5.js 模型
        const nn = window.ml5.neuralNetwork({
          task: "regression",
          debug: true,
          learningRate: 0.01,
          layers: [
            { type: "dense", units: 128, activation: "relu" },
            // { type: "dropout", rate: 0.2 },
            { type: "dense", units: 64, activation: "relu" },
            // { type: "dropout", rate: 0.2 },
            { type: "dense", units: 32, activation: "relu" },
            // { type: "dropout", rate: 0.2 },
            { type: "dense", units: 7, activation: "sigmoid" },
          ],
        });

        setModel(nn);
      }
    };

    initializeModel();
  }, []);

  useEffect(() => {
    const trainAndPredict = async () => {
      if (model) {
        // 数据预处理函数
        const preprocessData = (data: number[]) => {
          const redBalls = data.slice(0, 6).sort((a, b) => a - b);
          const blueBall = data[6];
          return [...redBalls, blueBall];
        };

        // 归一化函数
        const normalize = (value: number, max: number) => value / max;

        // 反归一化函数
        const denormalize = (value: number, max: number) =>
          Math.round(value * max);

        // 添加数据
        for (let i = historyData.length - 1; i >= 47; i--) {
          const inputs: number[] = [];
          for (let j = i; j > i - 47; j--) {
            const processedData = preprocessData([
              ...historyData[j].reds,
              historyData[j].blue,
            ]);
            inputs.push(
              ...processedData.slice(0, 6).map((x) => normalize(x, 33)),
              normalize(processedData[6], 16)
            );
          }
          const outputs = preprocessData([
            ...historyData[i - 47].reds,
            historyData[i - 47].blue,
          ]);
          const normalizedOutputs = [
            ...outputs.slice(0, 6).map((x) => normalize(x, 33)),
            normalize(outputs[6], 16),
          ];
          model.addData(inputs, normalizedOutputs);
        }

        // 调试信息
        console.log("Added data to model:", model.data);

        // 训练模型
        model.normalizeData();

        // Callback to check confidence and save the model if it exceeds the threshold

        const trainingOptions = {
          epochs: 200,
          batchSize: 32,

          callbacks: {
            onEpochEnd: async (epoch: number, logs: any) => {
              console.log(`Epoch ${epoch}: loss = ${logs.loss}`);
              if (logs.loss < 0.04) {
                await model.save("indexeddb://lottery-model");
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
        const last10Entries = historyData.slice(0, 47);
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
        await model.save("indexeddb://lottery-model");
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
          className="mt-4 px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 disabled:bg-gray-400"
        >
          {isSaving ? "Saving..." : "Save Model"}
        </button>
        {saveStatus && <p className="mt-2">{saveStatus}</p>}
      </div>
    </main>
  );
}
