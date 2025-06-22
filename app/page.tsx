"use client";
import { useEffect, useState } from "react";
import * as tf from "@tensorflow/tfjs";
import { Component } from "./show";
import { Card, CardContent, CardFooter } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { AppSidebar } from "@/components/app-sidebar";
import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbList,
  BreadcrumbPage,
  BreadcrumbSeparator,
} from "@/components/ui/breadcrumb";
import { Separator } from "@/components/ui/separator";
import {
  SidebarInset,
  SidebarProvider,
  SidebarTrigger,
} from "@/components/ui/sidebar";
import { useLotteryData } from "@/hooks/useLotteryData";

interface HistoryEntry {
  issue: string;
  reds: number[];
  blue: number;
}

interface ProphetPrediction {
  yhat: number;
  yhat_lower: number;
  yhat_upper: number;
}

interface ProphetPredictions {
  blue: ProphetPrediction;
  red: ProphetPrediction[];
}

interface PredictionResult {
  value: number;
}

interface ModelConfig {
  learningRate: number;
  layers: any[];
}

// Enhanced neural network configurations with 非理性优化 (Non-rational Optimization)
// 加入多种非理性因子：本福特定律、时间周期性、黄金比例、数字和谐性等
const modelConfigurations = [
  // 模型1: 本福特偏向网络 - 小数字优势
  {
    name: "本福特神经网络",
    learningRate: 0.001,
    nonRationalFactors: {
      benfordBias: 0.15, // 本福特定律权重
      timeCyclic: 0.08, // 时间周期性
      goldenRatio: 0.05, // 黄金比例调制
      harmonyBoost: 0.1, // 数字和谐性
      intuitionNoise: 0.12, // 直觉噪声
    },
    layers: [
      { type: "dense", units: 64, activation: "relu" },
      { type: "dropout", rate: 0.3 },
      { type: "dense", units: 32, activation: "relu" },
      { type: "dense", units: 7, activation: "linear" },
    ],
  },
  // 模型2: 时间魔法网络 - 周期性预测
  {
    name: "时间周期网络",
    learningRate: 0.0015,
    nonRationalFactors: {
      benfordBias: 0.08,
      timeCyclic: 0.2, // 强化时间因子
      goldenRatio: 0.1,
      harmonyBoost: 0.12,
      intuitionNoise: 0.1,
    },
    layers: [
      { type: "dense", units: 48, activation: "elu" },
      { type: "dropout", rate: 0.25 },
      { type: "dense", units: 24, activation: "elu" },
      { type: "dense", units: 7, activation: "linear" },
    ],
  },
  // 模型3: 黄金比例网络 - 自然和谐
  {
    name: "黄金比例网络",
    learningRate: 0.002,
    nonRationalFactors: {
      benfordBias: 0.1,
      timeCyclic: 0.05,
      goldenRatio: 0.18, // 强化黄金比例
      harmonyBoost: 0.15,
      intuitionNoise: 0.07,
    },
    layers: [
      { type: "dense", units: 32, activation: "relu" },
      { type: "dropout", rate: 0.4 },
      { type: "dense", units: 7, activation: "linear" },
    ],
  },
  // 模型4: 直觉网络 - 随机性与灵感
  {
    name: "直觉灵感网络",
    learningRate: 0.0018,
    nonRationalFactors: {
      benfordBias: 0.12,
      timeCyclic: 0.1,
      goldenRatio: 0.08,
      harmonyBoost: 0.1,
      intuitionNoise: 0.25, // 最高直觉噪声
    },
    layers: [
      { type: "dense", units: 56, activation: "tanh" },
      { type: "dropout", rate: 0.35 },
      { type: "dense", units: 28, activation: "relu" },
      { type: "dense", units: 7, activation: "linear" },
    ],
  },
  // 模型5: 数字和谐网络 - 相邻效应
  {
    name: "和谐共振网络",
    learningRate: 0.0012,
    nonRationalFactors: {
      benfordBias: 0.09,
      timeCyclic: 0.07,
      goldenRatio: 0.11,
      harmonyBoost: 0.22, // 最强和谐性
      intuitionNoise: 0.06,
    },
    layers: [
      { type: "dense", units: 40, activation: "relu" },
      { type: "dropout", rate: 0.28 },
      { type: "dense", units: 20, activation: "elu" },
      { type: "dense", units: 7, activation: "linear" },
    ],
  },
  // 模型6: 综合非理性网络 - 全因子融合
  {
    name: "非理性融合网络",
    learningRate: 0.0014,
    nonRationalFactors: {
      benfordBias: 0.13,
      timeCyclic: 0.12,
      goldenRatio: 0.12,
      harmonyBoost: 0.13,
      intuitionNoise: 0.15,
    },
    layers: [
      { type: "dense", units: 72, activation: "relu" },
      { type: "dropout", rate: 0.32 },
      { type: "dense", units: 36, activation: "elu" },
      { type: "dropout", rate: 0.2 },
      { type: "dense", units: 18, activation: "relu" },
      { type: "dense", units: 7, activation: "linear" },
    ],
  },
];

const prophetPredictions = {
  blue: { yhat: 1.14, yhat_lower: 2, yhat_upper: 15 },
  red: [
    { yhat: 2.27, yhat_lower: 1, yhat_upper: 14 },
    { yhat: 7.21, yhat_lower: 3, yhat_upper: 17 },
    { yhat: 10.27, yhat_lower: 7, yhat_upper: 23 },
    { yhat: 21.28, yhat_lower: 13, yhat_upper: 27 },
    { yhat: 23.23, yhat_lower: 18, yhat_upper: 32 },
    { yhat: 28.27, yhat_lower: 23, yhat_upper: 33 },
  ],
};

const standardizeData = (data: HistoryEntry[]): HistoryEntry[] => {
  return data.map((entry) => {
    const standardizedReds = entry.reds.map(
      (value, index) => value - prophetPredictions.red[index].yhat
    );
    const standardizedBlue = entry.blue - prophetPredictions.blue.yhat;
    return {
      issue: entry.issue,
      reds: standardizedReds,
      blue: standardizedBlue,
    };
  });
};

interface DenormalizedPrediction {
  reds: number[];
  blue: number;
}

const denormalizePrediction = (
  predictedValues: number[]
): DenormalizedPrediction => {
  const denormalizedReds: number[] = predictedValues
    .slice(0, 6)
    .map((value: number, index: number) =>
      Math.round(value + prophetPredictions.red[index].yhat)
    );
  const denormalizedBlue: number = Math.round(
    predictedValues[6] + prophetPredictions.blue.yhat
  );
  return { reds: denormalizedReds, blue: denormalizedBlue };
};

export default function Home() {
  const [modelInstances, setModelInstances] = useState<any[]>([]);
  const [modelPredictions, setModelPredictions] = useState<number[][]>([]);
  const [aggregatedPrediction, setAggregatedPrediction] = useState<
    number[] | null
  >(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [currentModelIndex, setCurrentModelIndex] = useState<number>(0);
  const [progress, setProgress] = useState<number>(0);
  const [show, setShow] = useState(true);
  const { historyData, isLoading, error } = useLotteryData();

  const trainNumber = 12;

  useEffect(() => {
    const initializeModels = async () => {
      if (typeof window !== "undefined" && window.ml5) {
        await tf.ready();

        const instances = await Promise.all(
          modelConfigurations.map((config) => {
            const nn = window.ml5.neuralNetwork({
              task: "regression",
              debug: true,
              learningRate: config.learningRate,
              layers: config.layers,
            });
            return nn;
          })
        );

        setModelInstances(instances);
      }
    };

    initializeModels();
  }, []);

  const trainAndPredictWithModel = async (
    modelIndex: number
  ): Promise<number[] | null> => {
    if (!modelInstances[modelIndex] || historyData.length === 0) return null;

    // Create a new model instance each time instead of trying to clear the old one
    const config = modelConfigurations[modelIndex];
    const model = window.ml5.neuralNetwork({
      task: "regression",
      debug: true,
      learningRate: config.learningRate,
      layers: config.layers,
    });

    const standardizedHistoryData = standardizeData(historyData);

    for (let i = standardizedHistoryData.length - 1; i >= trainNumber; i--) {
      let inputs = [];
      for (let j = i; j > i - trainNumber; j--) {
        let entry = standardizedHistoryData[j];
        inputs.push(...entry.reds, entry.blue);
      }

      const nextEntry = standardizedHistoryData[i - trainNumber];
      const outputs = [...nextEntry.reds, nextEntry.blue];
      model.addData(inputs, outputs);
    }

    // Rest of your function remains the same...
    const trainingOptions = {
      epochs: 228,
      batchSize: trainNumber,
      validationSplit: 0.2,
    };

    return new Promise((resolve, reject) => {
      model.train(trainingOptions, () => {
        const lastEntries = standardizeData(historyData.slice(0, trainNumber));
        const inputData = lastEntries.flatMap((entry) => [
          ...entry.reds,
          entry.blue,
        ]);
        model.predict(inputData, (results: PredictionResult[], err: any) => {
          if (err) {
            console.error(`Error in model ${modelIndex + 1}:`, err);
            reject(err);
          } else if (Array.isArray(results) && results.length === 7) {
            const standardizedPrediction = results.map((r) => r.value);
            console.log(
              `Model ${modelIndex + 1} 原始预测:`,
              standardizedPrediction
            );

            // 反标准化
            const denormalizedPrediction = denormalizePrediction(
              standardizedPrediction
            );
            let fullPrediction = [
              ...denormalizedPrediction.reds,
              denormalizedPrediction.blue,
            ];

            // 应用非理性优化 🔮✨
            const optimizedPrediction = applyNonRationalOptimization(
              fullPrediction,
              config
            ); // 确保预测值在合理范围内
            let finalPrediction = optimizedPrediction.map((value, index) => {
              const isBlue = index === 6;
              const maxNum = isBlue ? 16 : 33;
              return Math.max(1, Math.min(maxNum, Math.round(value)));
            });

            // 应用红球去重优化 🎯
            finalPrediction = ensureUniqueRedBalls(finalPrediction);

            console.log(
              `Model ${modelIndex + 1} (${config.name}) 非理性优化+去重后:`,
              finalPrediction
            );

            resolve(finalPrediction);
          } else {
            reject("Invalid prediction result");
          }
        });
      });
    });
  };

  const runAllModels = async () => {
    setShow(false);
    setLoading(true);
    setModelPredictions([]);
    setCurrentModelIndex(0);
    setProgress(0);

    const predictions: number[][] = [];

    for (let i = 0; i < modelConfigurations.length; i++) {
      setCurrentModelIndex(i);
      try {
        const prediction = await trainAndPredictWithModel(i);
        if (prediction) {
          predictions.push(prediction);
          setModelPredictions([...predictions]);
        }
        setProgress(((i + 1) / modelConfigurations.length) * 100);
      } catch (error) {
        console.error(`Error with model ${i + 1}:`, error);
      }
    }

    // Calculate aggregated prediction
    if (predictions.length > 0) {
      const aggregated = aggregatePredictions(predictions);
      setAggregatedPrediction(aggregated);
    }

    setLoading(false);
  };

  const aggregatePredictions = (predictions: number[][]): number[] => {
    // For each position, get the most frequent number
    const result = [];

    // Handle red balls (first 6 positions)
    for (let pos = 0; pos < 6; pos++) {
      const counters: Record<number, number> = {};

      // Count occurrences of each number in this position
      predictions.forEach((pred) => {
        const val = pred[pos];
        counters[val] = (counters[val] || 0) + 1;
      });

      // Find the most frequent number
      let maxCount = 0;
      let mostFrequent = 0;

      for (const [numStr, count] of Object.entries(counters)) {
        const num = parseInt(numStr);
        if (count > maxCount) {
          maxCount = count;
          mostFrequent = num;
        }
      }

      result.push(mostFrequent);
    }

    // Handle blue ball (last position)
    const blueCounters: Record<number, number> = {};
    predictions.forEach((pred) => {
      const blue = pred[6];
      blueCounters[blue] = (blueCounters[blue] || 0) + 1;
    });

    let maxBlueCount = 0;
    let mostFrequentBlue = 0;

    for (const [numStr, count] of Object.entries(blueCounters)) {
      const num = parseInt(numStr);
      if (count > maxBlueCount) {
        maxBlueCount = count;
        mostFrequentBlue = num;
      }
    }

    result.push(mostFrequentBlue);

    return result;
  };

  // 非理性优化函数集合 - 为ml5.js模型添加神秘学和直觉元素
  const nonRationalOptimizers = {
    // 本福特定律偏向 - 小数字更受青睐
    benfordBias: (value: number, position: number, isBlue: boolean): number => {
      const maxNum = isBlue ? 16 : 33;
      const normalizedValue = Math.max(1, Math.min(maxNum, Math.round(value)));
      // 本福特定律：P(d) = log10(1 + 1/d)
      const benfordFactor = Math.log10(1 + 1 / normalizedValue) / Math.log10(2);
      const adjustment = (benfordFactor - 0.5) * 3; // 放大影响
      return value + adjustment;
    },

    // 时间周期性魔法 - 基于当前时间的神秘调整
    timeCyclic: (value: number, position: number, isBlue: boolean): number => {
      const now = new Date();
      const dayOfWeek = now.getDay(); // 0-6
      const dayOfMonth = now.getDate(); // 1-31
      const hour = now.getHours(); // 0-23

      // 星期魔法：周日和周三被认为是幸运日
      const weeklyBoost = dayOfWeek === 0 || dayOfWeek === 3 ? 1.2 : 0.8;

      // 月份魔法：月初和月末有特殊能量
      const monthlyBoost = dayOfMonth <= 7 || dayOfMonth >= 25 ? 1.15 : 0.9;

      // 时辰魔法：黄金时间段(7-9点, 19-21点)
      const hourlyBoost =
        (hour >= 7 && hour <= 9) || (hour >= 19 && hour <= 21) ? 1.1 : 0.95;

      const timeAdjustment = (weeklyBoost * monthlyBoost * hourlyBoost - 1) * 2;
      return value + timeAdjustment;
    },

    // 黄金比例调制 - 使用φ(1.618...)进行自然和谐调整
    goldenRatio: (value: number, position: number, isBlue: boolean): number => {
      const φ = (1 + Math.sqrt(5)) / 2; // 黄金比例 ≈ 1.618
      const goldenWave = Math.sin(position * φ) * 0.618; // 使用黄金比例的倒数
      const harmonic = Math.cos(value / φ) * 0.382; // 另一个黄金比例相关值
      return value + goldenWave + harmonic;
    },

    // 数字和谐性 - 相邻数字的"引力"效应
    harmonyBoost: (
      value: number,
      position: number,
      isBlue: boolean,
      allValues?: number[]
    ): number => {
      if (!allValues) return value;

      const roundedValue = Math.round(value);
      let harmonyScore = 0;

      // 计算与其他预测值的"和谐度"
      allValues.forEach((otherValue, otherPos) => {
        if (otherPos !== position) {
          const distance = Math.abs(roundedValue - Math.round(otherValue));
          if (distance === 1) harmonyScore += 0.5; // 相邻数字加分
          if (distance === 7 || distance === 14) harmonyScore += 0.3; // 神秘间隔
          if (distance === 5 || distance === 10) harmonyScore += 0.2; // 五进制和谐
        }
      });

      // 数字本身的"美学"价值
      const digitalBeauty =
        roundedValue % 7 === 0 || roundedValue % 11 === 0 ? 0.3 : 0;

      return value + harmonyScore + digitalBeauty;
    },

    // 直觉噪声 - 模拟人类灵感的随机突发
    intuitionNoise: (
      value: number,
      position: number,
      isBlue: boolean
    ): number => {
      // 灵感突发：10%概率产生强烈直觉
      if (Math.random() < 0.1) {
        const inspiration = (Math.random() - 0.5) * 8; // 强烈的直觉调整
        return value + inspiration;
      }

      // 日常直觉：轻微的感性调整
      const gentleIntuition = (Math.random() - 0.5) * 2;

      // 第六感加成：某些位置更容易产生直觉
      const sixthSenseBoost = position === 2 || position === 5 ? 1.5 : 1.0;

      return value + gentleIntuition * sixthSenseBoost;
    },
  };

  // 应用非理性优化的函数
  const applyNonRationalOptimization = (
    predictions: number[],
    modelConfig: any
  ): number[] => {
    const factors = modelConfig.nonRationalFactors;
    if (!factors) return predictions;

    let optimizedPredictions = [...predictions];

    // 逐个应用各种非理性因子
    optimizedPredictions = optimizedPredictions.map((value, index) => {
      const isBlue = index === 6;
      let optimizedValue = value;

      // 应用本福特偏向
      if (factors.benfordBias > 0) {
        const benfordAdjustment = nonRationalOptimizers.benfordBias(
          value,
          index,
          isBlue
        );
        optimizedValue += (benfordAdjustment - value) * factors.benfordBias;
      }

      // 应用时间周期性
      if (factors.timeCyclic > 0) {
        const timeAdjustment = nonRationalOptimizers.timeCyclic(
          value,
          index,
          isBlue
        );
        optimizedValue += (timeAdjustment - value) * factors.timeCyclic;
      }

      // 应用黄金比例调制
      if (factors.goldenRatio > 0) {
        const goldenAdjustment = nonRationalOptimizers.goldenRatio(
          value,
          index,
          isBlue
        );
        optimizedValue += (goldenAdjustment - value) * factors.goldenRatio;
      }

      // 应用直觉噪声
      if (factors.intuitionNoise > 0) {
        const intuitionAdjustment = nonRationalOptimizers.intuitionNoise(
          value,
          index,
          isBlue
        );
        optimizedValue +=
          (intuitionAdjustment - value) * factors.intuitionNoise;
      }

      return optimizedValue;
    });

    // 最后应用数字和谐性（需要所有值一起计算）
    if (factors.harmonyBoost > 0) {
      optimizedPredictions = optimizedPredictions.map((value, index) => {
        const isBlue = index === 6;
        const harmonyAdjustment = nonRationalOptimizers.harmonyBoost(
          value,
          index,
          isBlue,
          optimizedPredictions
        );
        return value + (harmonyAdjustment - value) * factors.harmonyBoost;
      });
    }

    return optimizedPredictions;
  };

  // 红球去重和优化函数
  const ensureUniqueRedBalls = (prediction: number[]): number[] => {
    const redBalls = prediction.slice(0, 6);
    const blueBall = prediction[6];

    // 检查红球是否有重复
    const uniqueReds = Array.from(new Set(redBalls));

    if (uniqueReds.length === 6) {
      // 没有重复，直接返回排序后的结果
      return Array.from(uniqueReds.sort((a, b) => a - b)).concat([blueBall]);
    }

    // 有重复，需要替换重复的球
    const usedNumbers = new Set(uniqueReds);
    const finalReds = Array.from(uniqueReds);

    // 为缺少的球位选择新号码
    while (finalReds.length < 6) {
      // 使用非理性策略选择新号码
      let newNumber;

      // 30% 概率选择"幸运数字" (7, 11, 21, 28)
      if (Math.random() < 0.3) {
        const luckyNumbers = [7, 11, 21, 28].filter((n) => !usedNumbers.has(n));
        if (luckyNumbers.length > 0) {
          newNumber =
            luckyNumbers[Math.floor(Math.random() * luckyNumbers.length)];
        }
      }

      // 如果没有选到幸运数字，使用本福特倾向选择
      if (!newNumber) {
        const candidates = [];
        for (let i = 1; i <= 33; i++) {
          if (!usedNumbers.has(i)) {
            // 根据本福特定律加权
            const benfordWeight = Math.log10(1 + 1 / i);
            const weightedCount = Math.ceil(benfordWeight * 10);
            for (let j = 0; j < weightedCount; j++) {
              candidates.push(i);
            }
          }
        }

        if (candidates.length > 0) {
          newNumber = candidates[Math.floor(Math.random() * candidates.length)];
        } else {
          // 最后兜底：随机选择
          do {
            newNumber = Math.floor(Math.random() * 33) + 1;
          } while (usedNumbers.has(newNumber));
        }
      }

      finalReds.push(newNumber);
      usedNumbers.add(newNumber);
    }

    return Array.from(finalReds.sort((a, b) => a - b)).concat([blueBall]);
  };

  if (isLoading) {
    return (
      <div className="text-center w-full mt-20 text-blue-600 font-bold text-lg">
        Loading data...
      </div>
    );
  }

  if (error) {
    return (
      <div className="text-center w-full mt-20 text-rose-600 font-bold text-lg">
        Error: {error}
      </div>
    );
  }

  return (
    <>
      <header className="sticky top-0 flex shrink-0 items-center gap-2 border-b bg-background p-4">
        <SidebarTrigger className="-ml-1" />
        <Separator orientation="vertical" className="mr-2 h-4" />
        <Breadcrumb>
          <BreadcrumbList>
            <BreadcrumbItem className="hidden md:block">
              <BreadcrumbLink href="#">predict</BreadcrumbLink>
            </BreadcrumbItem>
            <BreadcrumbSeparator className="hidden md:block" />{" "}
            <BreadcrumbItem>
              <BreadcrumbPage>非理性神经网络预测 🔮</BreadcrumbPage>
            </BreadcrumbItem>
          </BreadcrumbList>
        </Breadcrumb>
      </header>
      <div className="flex flex-col h-full items-center justify-center">
        <CardContent className="w-full max-w-3xl">
          {loading ? (
            <div className="flex flex-col items-center gap-4">
              <div className="flex items-center">
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  fill="#7c3aed"
                  viewBox="0 0 256 256"
                  className="animate-spin w-6 h-6 mr-2"
                >
                  <path
                    d="M224,128a96,96,0,1,1-96-96A96,96,0,0,1,224,128Z"
                    opacity="0.2"
                  ></path>
                  <path d="M232,128a104,104,0,0,1-208,0c0-41,23.81-78.36,60.66-95.27a8,8,0,0,1,6.68,14.54C60.15,61.59,40,93.27,40,128a88,88,0,0,0,176,0c0-34.73-20.15-66.41-51.34-80.73a8,8,0,0,1,6.68-14.54C208.19,49.64,232,87,232,128Z"></path>
                </svg>{" "}
                <p>
                  正在训练{" "}
                  {modelConfigurations[currentModelIndex]?.name ||
                    `模型 ${currentModelIndex + 1}`}
                  ... ({modelConfigurations[currentModelIndex]?.layers.length}{" "}
                  层神经网络)
                  {Math.round(progress)}% 🧠✨
                </p>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2.5">
                <div
                  className="bg-blue-600 h-2.5 rounded-full"
                  style={{ width: `${progress}%` }}
                ></div>
              </div>
            </div>
          ) : (
            <>
              {" "}
              {aggregatedPrediction && (
                <div className="mb-8">
                  <h2 className="text-xl font-bold mb-4 text-center">
                    最终预测结果 (非理性神经网络) 🔮
                  </h2>
                  <p className="text-sm text-gray-600 text-center mb-4">
                    融合本福特定律、时间魔法、黄金比例、数字和谐性和直觉灵感的AI预测
                  </p>
                  <Component visitors={aggregatedPrediction} />
                </div>
              )}
              {modelPredictions.length > 0 && (
                <div>
                  <h3 className="text-lg font-bold mt-8 mb-4">
                    各非理性模型预测结果 ✨
                  </h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {modelPredictions.map((pred, idx) => (
                      <div
                        key={idx}
                        className="border rounded-md p-4 bg-gradient-to-br from-purple-50 to-blue-50"
                      >
                        <h4 className="font-medium mb-2 text-purple-700">
                          {modelConfigurations[idx]?.name || `模型 ${idx + 1}`}
                        </h4>
                        <div className="text-xs text-gray-500 mb-2">
                          {modelConfigurations[idx]?.nonRationalFactors && (
                            <div className="flex flex-wrap gap-1">
                              {modelConfigurations[idx].nonRationalFactors
                                .benfordBias > 0 && (
                                <span className="bg-blue-100 px-1 rounded">
                                  本福特
                                </span>
                              )}
                              {modelConfigurations[idx].nonRationalFactors
                                .timeCyclic > 0 && (
                                <span className="bg-green-100 px-1 rounded">
                                  时间魔法
                                </span>
                              )}
                              {modelConfigurations[idx].nonRationalFactors
                                .goldenRatio > 0 && (
                                <span className="bg-yellow-100 px-1 rounded">
                                  黄金比例
                                </span>
                              )}
                              {modelConfigurations[idx].nonRationalFactors
                                .harmonyBoost > 0 && (
                                <span className="bg-pink-100 px-1 rounded">
                                  数字和谐
                                </span>
                              )}
                              {modelConfigurations[idx].nonRationalFactors
                                .intuitionNoise > 0 && (
                                <span className="bg-purple-100 px-1 rounded">
                                  直觉灵感
                                </span>
                              )}
                            </div>
                          )}
                        </div>
                        <Component visitors={pred} />
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </>
          )}
        </CardContent>
        {show && (
          <CardFooter className="mt-8">
            {" "}
            <Button
              onClick={runAllModels}
              disabled={loading || modelInstances.length === 0}
              className="bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700 text-white"
            >
              🔮 开始非理性预测 ✨
            </Button>
          </CardFooter>
        )}
      </div>
    </>
  );
}
