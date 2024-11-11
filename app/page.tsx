"use client";
import { useEffect, useState } from "react";
import * as tf from "@tensorflow/tfjs";
import { historyData } from "./data/thisData";
import { Component } from "./show";
import { Card, CardContent, CardFooter } from "@/components/ui/card";
import { Button } from "@/components/ui/button";

// Subtracts yhat values (prophetPredictions) from reds and blue values in combined_data
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

// Define the interface for prediction results
interface PredictionResult {
  value: number;
}
// { issue: "24127", reds: [2, 5, 13, 20, 27, 32], blue: 10 },
// { issue: "24126", reds: [14, 18, 23, 24, 26, 33], blue: 10 },
const prophetPredictions = {
  blue: { yhat: 8, yhat_lower: 2, yhat_upper: 15 },
  red: [
    { yhat: 5, yhat_lower: 1, yhat_upper: 14 },
    { yhat: 10, yhat_lower: 3, yhat_upper: 18 },
    { yhat: 15, yhat_lower: 7, yhat_upper: 23 },
    { yhat: 20, yhat_lower: 13, yhat_upper: 27 },
    { yhat: 25, yhat_lower: 18, yhat_upper: 32 },
    { yhat: 30, yhat_lower: 23, yhat_upper: 33 },
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

// Denormalizes the prediction results by adding back the yhat values from prophetPredictions
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
              units: 256, // 增大初始层以捕获更多特征
              activation: "relu", // 使用ReLU避免梯度消失
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
              activation: "tanh", // 使用tanh激活函数
            },
            {
              type: "dense",
              units: 7, // 输出层保持7个单元
              activation: "linear", // 因为是回归问题用linear
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
        // Standardize history data
        const standardizedHistoryData = standardizeData(historyData);

        // Create inputs and outputs using standardized data
        for (
          let i = standardizedHistoryData.length - 1;
          i >= trainNumber;
          i--
        ) {
          let inputs = [];
          for (let j = i; j > i - trainNumber; j--) {
            let entry = standardizedHistoryData[j];
            inputs.push(...entry.reds, entry.blue);
          }

          // Define outputs with the next entry as the expected output
          const nextEntry = standardizedHistoryData[i - trainNumber];
          const outputs = [...nextEntry.reds, nextEntry.blue];
          model.addData(inputs, outputs);
        }

        const trainingOptions = {
          epochs: 270,
          batchSize: trainNumber,
          validationSplit: 0.2,
        };

        const finishedTraining = () => {
          console.log("Model trained!");
          makePrediction();
        };

        await model.train(trainingOptions, finishedTraining);
      }
    };

    const makePrediction = () => {
      const lastEntries = standardizeData(historyData.slice(0, trainNumber));
      const inputData = lastEntries.flatMap((entry) => [
        ...entry.reds,
        entry.blue,
      ]);

      model.predict(inputData, (results: PredictionResult[], err: any) => {
        if (err) {
          console.error(err);
        } else if (Array.isArray(results) && results.length === 7) {
          const standardizedPrediction: number[] = results.map(
            (r: PredictionResult) => r.value
          );
          const denormalizedPrediction: DenormalizedPrediction =
            denormalizePrediction(standardizedPrediction);

          console.log(
            "Standardized Prediction (differences):",
            standardizedPrediction
          );
          console.log(
            "Denormalized Prediction (final reds and blue):",
            denormalizedPrediction
          );
          const predictedValuesExceedBounds = standardizedPrediction.some(
            (value, index) => {
              const isRed = index < 6;
              const yhat_lower = isRed
                ? prophetPredictions.red[index].yhat_lower
                : prophetPredictions.blue.yhat_lower;
              const yhat_upper = isRed
                ? prophetPredictions.red[index].yhat_upper
                : prophetPredictions.blue.yhat_upper;
              const denormalizedValue =
                value +
                (isRed
                  ? prophetPredictions.red[index].yhat
                  : prophetPredictions.blue.yhat);
              return (
                denormalizedValue < yhat_lower || denormalizedValue > yhat_upper
              );
            }
          );

          if (predictedValuesExceedBounds) {
            console.log("训练失败");
          }

          setPrediction(
            denormalizedPrediction.reds.concat(denormalizedPrediction.blue)
          );
        }
      });
    };

    trainAndPredict();
  }, [model]);

  // Function to save the trained model
  const saveModel = async () => {
    setIsSaving(true);
    setSaveStatus("Saving model...");
    try {
      await model.save(); // Save the model in browser's local storage
      setSaveStatus("Model saved successfully!");
    } catch (error) {
      console.error("Error saving model:", error);
      setSaveStatus("Error saving model.");
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <Card className="flex justify-center items-center h-screen flex-col">
      <CardContent>
        {prediction && (
          <div className="h-3/5">
            <Component visitors={prediction ? prediction.slice(0, 7) : []} />
          </div>
        )}
      </CardContent>

      {/* Save Button */}

      <CardFooter>
        <Button onClick={saveModel} disabled={isSaving}>
          {prediction ? (
            <>
              {isSaving ? "Saving..." : "Save Model"}
              {saveStatus && <p>{saveStatus}</p>}
            </>
          ) : (
            <>
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
              <p>loading...</p>
            </>
          )}
        </Button>
      </CardFooter>
    </Card>
  );
}
