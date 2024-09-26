"use client";
import { useEffect, useState } from "react";
import * as tf from "@tensorflow/tfjs";

import { historyData } from "./data/historyData";
import ModeToggle from "@/components/mode-toggle";

// 添加 Prophet 预测结果
const prophetPredictions = {
  blue: { yhat: 8, yhat_lower: 2, yhat_upper: 14 },
  red: [
    { yhat: 4, yhat_lower: 1, yhat_upper: 9 },
    { yhat: 9, yhat_lower: 2, yhat_upper: 15 },
    { yhat: 14, yhat_lower: 8, yhat_upper: 21 },
    { yhat: 20, yhat_lower: 13, yhat_upper: 26 },
    { yhat: 24, yhat_lower: 18, yhat_upper: 30 },
    { yhat: 29, yhat_lower: 24, yhat_upper: 33 },
  ],
};

const historicalWeight = 0.7; // 历史数据权重
const prophetWeight = 0.3; // Prophet 预测结果权重
const blueBallWeightFactor = 33 / 16; // 蓝球与红球比例的缩放因子

export default function Home() {
  const [model, setModel] = useState<any>(null);
  const [prediction, setPrediction] = useState<number[] | null>(null);
  const [isSaving, setIsSaving] = useState<boolean>(false);
  const [saveStatus, setSaveStatus] = useState<string>("");

  const trainNumber = 32; // 用于训练的历史数据数量

  useEffect(() => {
    const initializeModel = async () => {
      if (typeof window !== "undefined" && window.ml5) {
        await tf.ready();

        // 创建 ml5.js 模型
        const nn = window.ml5.neuralNetwork({
          task: "regression",
          debug: true,
          learningRate: 0.001,
          layers: [
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
              units: 7, // 输出 7 个数值 (6 个红球 + 1 个蓝球)
              activation: "linear",
            },
          ],
        });

        setModel(nn);
      }
    };

    initializeModel();
  }, []);

  // Z-score 标准化函数
  const zScoreNormalize = (data: number[]) => {
    const mean = data.reduce((sum, val) => sum + val, 0) / data.length;
    const std = Math.sqrt(
      data.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / data.length
    );
    return data.map((val) => (val - mean) / (std || 1)); // 避免除以零
  };

  // Z-score 反标准化函数
  const zScoreDenormalize = (
    normalizedValue: number,
    originalData: number[]
  ) => {
    const mean =
      originalData.reduce((sum, val) => sum + val, 0) / originalData.length;
    const std = Math.sqrt(
      originalData.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) /
        originalData.length
    );
    return normalizedValue * std + mean;
  };

  useEffect(() => {
    const trainAndPredict = async () => {
      if (model) {
        const preprocessData = (data: number[]) => {
          const redBalls = data.slice(0, 6).sort((a, b) => a - b);
          const blueBall = data[6];
          return [...redBalls, blueBall];
        };

        // 获取所有红球和蓝球数据用于标准化
        const allRedBalls = historyData.flatMap((entry) => entry.reds);
        const allBlueBalls = historyData.map((entry) => entry.blue);

        // 添加数据
        for (let i = historyData.length - 1; i >= trainNumber; i--) {
          let inputs = [];
          for (let j = i; j > i - trainNumber; j--) {
            let processedData = preprocessData([
              ...historyData[j].reds,
              historyData[j].blue,
            ]);

            // 使用 Z-score 标准化
            let adjustedData = zScoreNormalize(processedData.slice(0, 6));
            let adjustedBlueBall = zScoreNormalize([processedData[6]])[0];

            // 添加 Prophet 预测结果作为特征
            const prophetRedFeatures = zScoreNormalize(
              prophetPredictions.red.map((pred) => pred.yhat)
            );
            const prophetBlueFeature = zScoreNormalize([
              prophetPredictions.blue.yhat,
            ])[0];

            // 使用加权平均组合历史数据和 Prophet 预测结果
            inputs.push(
              ...adjustedData.map((val) => val * historicalWeight), // 对历史红球数据施加权重
              adjustedBlueBall * historicalWeight * blueBallWeightFactor, // 对蓝球数据施加权重并缩放
              ...prophetRedFeatures.map((val) => val * prophetWeight), // 对 Prophet 红球施加权重
              prophetBlueFeature * prophetWeight * blueBallWeightFactor // 对 Prophet 蓝球施加权重并缩放
            );
          }

          let outputs = preprocessData([
            ...historyData[i - trainNumber].reds,
            historyData[i - trainNumber].blue,
          ]);

          // 对输出也使用 Z-score 标准化
          let adjustedOutputs = zScoreNormalize(outputs.slice(0, 6));
          let adjustedBlueOutput = zScoreNormalize([outputs[6]])[0];

          // 添加调整后的输出数据
          model.addData(inputs, [...adjustedOutputs, adjustedBlueOutput]);
        }

        const trainingOptions = {
          epochs: 100,
          batchSize: 32,
          shuffle: true,
          validationSplit: 0.2,
        };

        const finishedTraining = () => {
          console.log("Model trained!");
          makePrediction(preprocessData, allRedBalls, allBlueBalls);
        };

        await model.train(trainingOptions, finishedTraining);
      }
    };

    const makePrediction = (
      preprocessData: (data: number[]) => number[],
      allRedBalls: number[],
      allBlueBalls: number[]
    ) => {
      // 进行预测
      const lastEntries = historyData.slice(0, trainNumber);
      const inputData = lastEntries.flatMap((entry) => {
        const processedData = preprocessData([...entry.reds, entry.blue]);

        // 分别处理红球和蓝球
        const normalizedRedBalls = zScoreNormalize(processedData.slice(0, 6)); // 处理红球数据
        const normalizedBlueBall = zScoreNormalize([processedData[6]])[0]; // 处理蓝球数据

        // 添加 Prophet 预测结果作为特征
        const prophetRedFeatures = zScoreNormalize(
          prophetPredictions.red.map((pred) => pred.yhat)
        );
        const prophetBlueFeature = zScoreNormalize([
          prophetPredictions.blue.yhat,
        ])[0];

        // 使用加权平均组合历史数据和 Prophet 预测结果
        const weightedRedBalls = normalizedRedBalls.map(
          (val) => val * historicalWeight
        ); // 对历史红球施加权重
        const weightedBlueBall =
          normalizedBlueBall * historicalWeight * blueBallWeightFactor; // 对历史蓝球施加权重并缩放

        const weightedProphetRedBalls = prophetRedFeatures.map(
          (val) => val * prophetWeight
        ); // 对 Prophet 红球施加权重
        const weightedProphetBlueBall =
          prophetBlueFeature * prophetWeight * blueBallWeightFactor; // 对 Prophet 蓝球施加权重并缩放

        // 将红球和蓝球的数据分别合并
        return [
          ...weightedRedBalls, // 历史红球
          weightedBlueBall, // 历史蓝球
          ...weightedProphetRedBalls, // Prophet 红球预测
          weightedProphetBlueBall, // Prophet 蓝球预测
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
                value = Math.round(zScoreDenormalize(r.value, allRedBalls));
                console.log("Red ball value:", value);
                const prophetPred = prophetPredictions.red[index];
                // 确保预测值在 Prophet 模型的置信区间内
                value = Math.max(
                  Math.min(value, prophetPred.yhat_upper),
                  prophetPred.yhat_lower
                );
              } else {
                // 蓝球
                value = Math.round(zScoreDenormalize(r.value, allBlueBalls));
                console.log("Blue ball value:", value);
                // 确保预测值在 Prophet 模型的置信区间内
                value = Math.max(
                  Math.min(value, prophetPredictions.blue.yhat_upper),
                  prophetPredictions.blue.yhat_lower
                );
              }
              return Math.max(
                1,
                Math.min(index < 6 ? 33 : 16, Math.round(value))
              );
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
