"use client";
import { useEffect, useState } from "react";
import * as tf from "@tensorflow/tfjs";
import { historyData } from "./data/thisData";
import { Component } from "./show";

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

// Adjust weights – give more importance to historical data initially
const historicalWeight = 0.2;
const prophetWeight = 0.8;
const blueBallWeightFactor = 33 / 16;

export default function Home() {
  const [model, setModel] = useState<any>(null);
  const [prediction, setPrediction] = useState<number[] | null>(null);
  const [isSaving, setIsSaving] = useState<boolean>(false);
  const [saveStatus, setSaveStatus] = useState<string>("");

  const trainNumber = 16; // 用于训练的历史数据数量

  useEffect(() => {
    const initializeModel = async () => {
      if (typeof window !== "undefined" && window.ml5) {
        await tf.ready();

        // 创建 ml5.js 模型
        const nn = window.ml5.neuralNetwork({
          task: "regression",
          debug: true,
          learningRate: 0.0001,
          layers: [
            {
              type: "dense",
              units: 128,
              activation: "relu", // 改为ReLU
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
              activation: "linear", // 输出层保持线性激活
            },
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
        const preprocessData = (data: number[]) => {
          const redBalls = data.slice(0, 6).sort((a, b) => a - b);
          const blueBall = data[6];
          return [...redBalls, blueBall];
        };

        // Separate normalization for red and blue balls
        const allRedBalls = historyData.flatMap((entry) => entry.reds);
        const allBlueBalls = historyData.map((entry) => entry.blue);
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

        const normalizeRed = (x: number) => (x - redMean) / (redStd || 1);
        const normalizeBlue = (x: number) => (x - blueMean) / (blueStd || 1);
        const denormalizeRed = (x: number) => x * (redStd || 1) + redMean;
        const denormalizeBlue = (x: number) => x * (blueStd || 1) + blueMean;

        // Add data with improved feature engineering
        for (let i = historyData.length - 1; i >= trainNumber; i--) {
          let inputs = [];
          for (let j = i; j > i - trainNumber; j--) {
            let processedData = preprocessData([
              ...historyData[j].reds,
              historyData[j].blue,
            ]);

            // Separate normalization for red and blue
            let adjustedRedData = processedData.slice(0, 6).map(normalizeRed);
            let adjustedBlueBall = normalizeBlue(processedData[6]);

            // Prophet features – use differences instead of raw values
            const prophetRedDiffs = prophetPredictions.red.map((pred, index) =>
              index === 0
                ? pred.yhat
                : pred.yhat - prophetPredictions.red[index - 1].yhat
            );
            const prophetRedFeatures = prophetRedDiffs.map(normalizeRed); // Normalize differences
            const prophetBlueFeature = normalizeBlue(
              prophetPredictions.blue.yhat
            ); // Normalize

            inputs.push(
              ...adjustedRedData.map((val) => val * historicalWeight),
              adjustedBlueBall * historicalWeight * blueBallWeightFactor,
              ...prophetRedFeatures.map((val) => val * prophetWeight),
              prophetBlueFeature * prophetWeight * blueBallWeightFactor
            );
          }

          let outputs = preprocessData([
            ...historyData[i - trainNumber].reds,
            historyData[i - trainNumber].blue,
          ]);
          let adjustedOutputs = outputs.slice(0, 6).map(normalizeRed);
          let adjustedBlueOutput = normalizeBlue(outputs[6]);
          model.addData(inputs, [...adjustedOutputs, adjustedBlueOutput]);
        }

        const trainingOptions = {
          epochs: 150,
          batchSize: 16,
          validationSplit: 0.2,
        };

        const finishedTraining = () => {
          console.log("Model trained!");
          makePrediction(
            preprocessData,
            normalizeRed,
            normalizeBlue,
            denormalizeRed,
            denormalizeBlue
          );
        };

        await model.train(trainingOptions, finishedTraining);
      }
    };

    const makePrediction = (
      preprocessData: (data: number[]) => number[],
      normalizeRed: (x: number) => number,
      normalizeBlue: (x: number) => number,
      denormalizeRed: (x: number) => number,
      denormalizeBlue: (x: number) => number
    ) => {
      // 进行预测
      const lastEntries = historyData.slice(0, trainNumber);
      const inputData = lastEntries.flatMap((entry) => {
        const processedData = preprocessData([...entry.reds, entry.blue]);

        // 分别处理红球和蓝球
        const normalizedRedBalls = processedData.slice(0, 6).map(normalizeRed); // 处理红球数据
        const normalizedBlueBall = normalizeBlue(processedData[6]); // 处理蓝球数据

        // 添加 Prophet 预测结果作为特征
        const prophetRedDiffs = prophetPredictions.red.map((pred, index) =>
          index === 0
            ? pred.yhat
            : pred.yhat - prophetPredictions.red[index - 1].yhat
        );
        const prophetRedFeatures = prophetRedDiffs.map(normalizeRed);
        const prophetBlueFeature = normalizeBlue(prophetPredictions.blue.yhat);

        // 使用加权平均组合历史数据和 Prophet 预测结果
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
      });

      model.predict(inputData, (results: any, err: any) => {
        if (err) {
          console.error(err, "something went wrong");
        } else {
          if (Array.isArray(results) && results.length === 7) {
            const adjustedPrediction = results.map((r, index) => {
              let value;
              if (index < 6) {
                value = Math.round(denormalizeRed(r.value));
                console.log("Red ball value:", value);
                const prophetPred = prophetPredictions.red[index];
                // 确保预测值在 Prophet 模型的置信区间内
                value = Math.max(
                  Math.min(value, prophetPred.yhat_upper),
                  prophetPred.yhat_lower
                );
              } else {
                value = Math.round(denormalizeBlue(r.value));
                console.log("Blue ball value:", value);
                // 确保预测值在 Prophet 模型的置信区间内
                value = Math.max(
                  Math.min(value, prophetPredictions.blue.yhat_upper),
                  prophetPredictions.blue.yhat_lower
                );
              }
              // Ensure values are within valid ranges
              return Math.max(1, Math.min(index < 6 ? 33 : 16, value));
            });
            setPrediction(adjustedPrediction); // Update state with prediction
            console.log("Prediction:", adjustedPrediction);
          }
        }
      });
    };

    trainAndPredict();
  }, [model]);

  return (
    <div className="flex justify-center items-center h-screen">
      {prediction && (
        <div className="h-3/5">
          {/* <h2>Predicted Numbers:</h2>
          <p>Red: {prediction.slice(0, 6).join(", ")}</p>
          <p>Blue: {prediction[6]}</p> */}
          <Component visitors={prediction ? prediction.slice(0, 7) : []} />
        </div>
      )}
    </div>
  );
}
