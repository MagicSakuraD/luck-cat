"use client";
import { useEffect, useState } from "react";
import * as tf from "@tensorflow/tfjs";
// import { historyData } from "./data/thisData";
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

const prophetPredictions = {
  blue: { yhat: 8.55, yhat_lower: 2, yhat_upper: 15 },
  red: [
    { yhat: 3.42, yhat_lower: 1, yhat_upper: 14 },
    { yhat: 8.28, yhat_lower: 3, yhat_upper: 17 },
    { yhat: 15.49, yhat_lower: 7, yhat_upper: 23 },
    { yhat: 20.06, yhat_lower: 13, yhat_upper: 27 },
    { yhat: 24.02, yhat_lower: 18, yhat_upper: 32 },
    { yhat: 28.89, yhat_lower: 23, yhat_upper: 33 },
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
  const [model, setModel] = useState<any>(null);
  const [prediction, setPrediction] = useState<number[] | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [show, setShow] = useState(true);
  const { historyData, isLoading, error } = useLotteryData();

  const trainNumber = 21;

  useEffect(() => {
    const initializeModel = async () => {
      if (typeof window !== "undefined" && window.ml5) {
        await tf.ready();

        const nn = window.ml5.neuralNetwork({
          task: "regression",
          debug: true,
          learningRate: 0.001,
          layers: [
            {
              type: "dense",
              units: 512,
              activation: "elu", // 处理初始标准化数据
            },
            {
              type: "dense",
              units: 256,
              activation: "elu", // 继续特征提取
            },
            {
              type: "dense",
              units: 128,
              activation: "tanh", // 控制数值范围
            },
            {
              type: "dropout",
              rate: 0.2, // 防止过拟合
            },
            {
              type: "dense",
              units: 64,
              activation: "tanh", // 进一步处理
            },
            {
              type: "dense",
              units: 7,
              activation: "linear", // 输出层，预测偏差值
            },
          ],
        });

        setModel(nn);
      }
    };

    initializeModel();
  }, []);

  const trainAndPredict = async () => {
    setShow(false);
    if (model && historyData.length > 0) {
      setLoading(true);
      // console.log("thisdata", thisData);
      console.log(historyData);
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

      const trainingOptions = {
        epochs: 210,
        batchSize: trainNumber,
        validationSplit: 0.2,
      };

      await model.train(trainingOptions, () => {
        const lastEntries = standardizeData(historyData.slice(0, trainNumber));
        const inputData = lastEntries.flatMap((entry) => [
          ...entry.reds,
          entry.blue,
        ]);

        model.predict(inputData, (results: PredictionResult[], err: any) => {
          setLoading(false);
          if (err) {
            console.error(err);
          } else if (Array.isArray(results) && results.length === 7) {
            const standardizedPrediction = results.map((r) => r.value);
            console.log("Standardized Prediction:", standardizedPrediction);
            const denormalizedPrediction = denormalizePrediction(
              standardizedPrediction
            );
            setPrediction(
              denormalizedPrediction.reds.concat(denormalizedPrediction.blue)
            );
          }
        });
      });
    }
  };

  if (isLoading) {
    return <div>Loading data...</div>;
  }

  if (error) {
    return <div>Error: {error}</div>;
  }

  return (
    <>
      <header className="sticky top-0 flex shrink-0 items-center gap-2 border-b bg-background p-4">
        <SidebarTrigger className="-ml-1" />
        <Separator orientation="vertical" className="mr-2 h-4" />
        <Breadcrumb>
          <BreadcrumbList>
            <BreadcrumbItem className="hidden md:block">
              <BreadcrumbLink href="#">All Inboxes</BreadcrumbLink>
            </BreadcrumbItem>
            <BreadcrumbSeparator className="hidden md:block" />
            <BreadcrumbItem>
              <BreadcrumbPage>Inbox</BreadcrumbPage>
            </BreadcrumbItem>
          </BreadcrumbList>
        </Breadcrumb>
      </header>
      <div className="flex h-full items-center justify-center">
        <CardContent>
          {loading ? (
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
              <p>推理中...</p>
            </div>
          ) : (
            prediction && (
              <div className="h-3/5">
                <Component
                  visitors={prediction ? prediction.slice(0, 7) : []}
                />
              </div>
            )
          )}
        </CardContent>
        {show && (
          <CardFooter className="mt-28">
            <Button onClick={trainAndPredict} disabled={loading}>
              开始
            </Button>
          </CardFooter>
        )}
      </div>
    </>
  );
}
