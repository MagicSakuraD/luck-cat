"use client";
import { useState, useEffect } from "react";
import * as tf from "@tensorflow/tfjs";

import { thisData } from "../data/thisData";

export default function PredictionPage() {
  const [model, setModel] = useState<any>(null);
  const [prediction, setPrediction] = useState<number[] | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const loadModelAndPredict = async () => {
      if (typeof window !== "undefined" && window.ml5) {
        try {
          // 确保 TensorFlow.js 已经初始化
          await tf.ready();

          // 创建一个新的神经网络实例
          const nn = window.ml5.neuralNetwork({
            task: "regression",
            debug: true,
            // learningRate: 0.07,
            // hiddenUnits: 64,
          });

          // // 加载本地保存的模型
          // const modelPath = "/model/indexeddb___lottery-model.json"; // 本地模型的相对路径
          // 使用 JSON 对象指定模型文件路径
          const modelInfo = {
            model: "/model/model.json",
            metadata: "/model/model_meta.json",
            weights: "/model/model.weights.bin",
          };

          nn.load(modelInfo, () => {
            console.log("Model loaded successfully");
            setModel(nn);
            makePrediction(nn);
          });
        } catch (err) {
          console.error("Error initializing:", err);
          setError("Error initializing model");
          setIsLoading(false);
        }
      }
    };

    loadModelAndPredict();
  }, []);

  const makePrediction = (loadedModel: any) => {
    // 准备输入数据
    const lastEntries = thisData.slice(0, 36);
    const inputData = lastEntries.flatMap((entry) => {
      const processedData = preprocessData([...entry.reds, entry.blue]);
      return [
        ...processedData.slice(0, 6).map((x) => normalize(x, 33)),
        normalize(processedData[6], 16),
      ];
    });

    // 进行预测
    loadedModel.predict(inputData, (results: any, err: any) => {
      if (err) {
        console.error(err);
        setError("Error making prediction");
      } else {
        console.log("Raw prediction results:", results);

        if (Array.isArray(results) && results.length === 7) {
          const adjustedPrediction = results.map((r: any, index: number) => {
            if (index < 6) {
              return Math.max(1, Math.min(33, denormalize(r.value, 33)));
            } else {
              return Math.max(1, Math.min(16, denormalize(r.value, 16)));
            }
          });

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
          setError("Prediction did not return 7 numbers");
        }
      }
      setIsLoading(false);
    });
  };

  // 辅助函数
  const preprocessData = (data: number[]) => {
    const redBalls = data.slice(0, 6).sort((a, b) => a - b);
    const blueBall = data[6];
    return [...redBalls, blueBall];
  };

  const normalize = (value: number, max: number) => value / max;

  const denormalize = (value: number, max: number) => Math.round(value * max);

  return (
    <main className="flex min-h-screen flex-col items-center justify-between p-24">
      <div>
        <h1 className="text-2xl font-bold mb-4">Prediction</h1>
        {isLoading ? (
          <p>Loading model and making prediction...</p>
        ) : error ? (
          <p className="text-red-500">{error}</p>
        ) : prediction ? (
          <div>
            <p className="mb-2">Predicted numbers:</p>
            <p>Red balls: {prediction.slice(0, 6).join(", ")}</p>
            <p>Blue ball: {prediction[6]}</p>
          </div>
        ) : (
          <p>No prediction available</p>
        )}
      </div>
    </main>
  );
}
