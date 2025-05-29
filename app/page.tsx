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

// Optimized neural network configurations for lottery prediction
// Note: Neural networks are not ideal for truly random lottery prediction
const modelConfigurations = [
  // 模型1: 简化ReLU网络 - 一致激活函数
  {
    learningRate: 0.001,
    layers: [
      { type: "dense", units: 64, activation: "relu" },
      { type: "dropout", rate: 0.3 },
      { type: "dense", units: 32, activation: "relu" },
      { type: "dense", units: 7, activation: "linear" },
    ],
  },
  // 模型2: ELU激活 - 自归一化特性
  {
    learningRate: 0.0015,
    layers: [
      { type: "dense", units: 48, activation: "elu" },
      { type: "dropout", rate: 0.25 },
      { type: "dense", units: 24, activation: "elu" },
      { type: "dense", units: 7, activation: "linear" },
    ],
  },
  // 模型3: 浅层网络 - 防止过拟合
  {
    learningRate: 0.002,
    layers: [
      { type: "dense", units: 32, activation: "relu" },
      { type: "dropout", rate: 0.4 },
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
              `Model ${modelIndex + 1} Prediction:`,
              standardizedPrediction
            );
            const denormalizedPrediction = denormalizePrediction(
              standardizedPrediction
            );
            const fullPrediction = [
              ...denormalizedPrediction.reds,
              denormalizedPrediction.blue,
            ];
            resolve(fullPrediction);
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
            <BreadcrumbSeparator className="hidden md:block" />
            <BreadcrumbItem>
              <BreadcrumbPage>prophetPredictions</BreadcrumbPage>
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
                </svg>
                <p>
                  正在训练模型 {currentModelIndex + 1}/6:{" "}
                  {modelConfigurations[currentModelIndex]?.layers.length} 层 (
                  {Math.round(progress)}%)
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
              {aggregatedPrediction && (
                <div className="mb-8">
                  <h2 className="text-xl font-bold mb-4 text-center">
                    最终预测结果
                  </h2>
                  <Component visitors={aggregatedPrediction} />
                </div>
              )}

              {modelPredictions.length > 0 && (
                <div>
                  <h3 className="text-lg font-bold mt-8 mb-4">
                    各模型预测结果
                  </h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {modelPredictions.map((pred, idx) => (
                      <div key={idx} className="border rounded-md p-4">
                        <h4 className="font-medium mb-2">模型 {idx + 1}</h4>
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
            <Button
              onClick={runAllModels}
              disabled={loading || modelInstances.length === 0}
            >
              开始预测
            </Button>
          </CardFooter>
        )}
      </div>
    </>
  );
}
