"use client";
import { useEffect, useState } from "react";
import * as tf from "@tensorflow/tfjs";

// 假设这是最近 50 期的数据
const historyData = [
  { issue: "1024085", reds: [1, 5, 10, 21, 23, 27], blue: 10 },
  { issue: "1024084", reds: [1, 8, 10, 13, 19, 29], blue: 13 },
  { issue: "1024083", reds: [3, 9, 14, 29, 32, 33], blue: 10 },
  { issue: "1024082", reds: [1, 2, 13, 16, 17, 29], blue: 16 },
  { issue: "1024081", reds: [1, 6, 12, 17, 23, 25], blue: 4 },
  { issue: "1024080", reds: [6, 11, 12, 27, 29, 10], blue: 13 },
  { issue: "1024079", reds: [2, 3, 6, 7, 16, 26], blue: 4 },
  { issue: "1024078", reds: [5, 9, 14, 21, 22, 26], blue: 12 },
  { issue: "1024077", reds: [1, 4, 6, 14, 17, 22], blue: 8 },
  { issue: "1024076", reds: [3, 22, 24, 27, 29, 32], blue: 10 },
  { issue: "1024075", reds: [3, 5, 8, 18, 22, 28], blue: 1 },
  { issue: "1024074", reds: [7, 8, 10, 22, 24, 32], blue: 7 },
  { issue: "1024073", reds: [3, 9, 12, 18, 28, 10], blue: 1 },
  { issue: "1024072", reds: [1, 14, 10, 21, 23, 27], blue: 6 },
  { issue: "1024071", reds: [2, 8, 19, 28, 10, 31], blue: 14 },
  { issue: "1024070", reds: [4, 13, 18, 10, 22, 28], blue: 5 },
  { issue: "1024069", reds: [6, 13, 10, 21, 24, 32], blue: 6 },
  { issue: "1024068", reds: [1, 2, 7, 19, 10, 21], blue: 1 },
  { issue: "1024067", reds: [1, 11, 13, 17, 25, 29], blue: 3 },
  { issue: "1024066", reds: [8, 9, 12, 22, 26, 32], blue: 13 },
  { issue: "1024065", reds: [3, 5, 9, 10, 19, 22], blue: 14 },
  { issue: "1024064", reds: [6, 8, 17, 18, 28, 10], blue: 5 },
  { issue: "1024063", reds: [7, 14, 16, 23, 28, 32], blue: 4 },
  { issue: "1024062", reds: [1, 7, 10, 16, 18, 27], blue: 16 },
  { issue: "1024061", reds: [1, 9, 18, 22, 25, 28], blue: 2 },
  { issue: "1024060", reds: [1, 2, 6, 10, 22, 28], blue: 10 },
  { issue: "1024059", reds: [1, 3, 14, 25, 31, 33], blue: 7 },
  { issue: "1024058", reds: [8, 12, 13, 17, 27, 29], blue: 13 },
  { issue: "1024057", reds: [3, 8, 17, 18, 23, 31], blue: 8 },
  { issue: "1024056", reds: [5, 18, 10, 24, 25, 26], blue: 6 },
  { issue: "1024055", reds: [2, 4, 6, 7, 16, 29], blue: 3 },
  { issue: "1024054", reds: [8, 13, 10, 25, 31, 32], blue: 3 },
  { issue: "1024053", reds: [7, 14, 21, 22, 28, 33], blue: 7 },
  { issue: "1024052", reds: [7, 10, 11, 10, 17, 21], blue: 3 },
  { issue: "1024051", reds: [5, 9, 13, 10, 23, 28], blue: 6 },
  { issue: "1024050", reds: [1, 3, 7, 10, 22, 33], blue: 2 },
  { issue: "1024049", reds: [12, 10, 17, 23, 26, 32], blue: 11 },
  { issue: "1024048", reds: [2, 9, 10, 19, 26, 28], blue: 2 },
  { issue: "1024047", reds: [7, 8, 21, 26, 29, 10], blue: 10 },
  { issue: "1024046", reds: [2, 6, 10, 11, 17, 29], blue: 10 },
  { issue: "1024045", reds: [2, 8, 19, 23, 24, 26], blue: 3 },
  { issue: "1024044", reds: [2, 6, 17, 25, 32, 33], blue: 6 },
  { issue: "1024043", reds: [4, 6, 7, 14, 10, 24], blue: 8 },
  { issue: "1024042", reds: [2, 4, 5, 14, 26, 32], blue: 14 },
  { issue: "1024041", reds: [2, 9, 12, 22, 25, 33], blue: 16 },
  { issue: "1024040", reds: [11, 14, 18, 19, 23, 26], blue: 2 },
  { issue: "1024039", reds: [2, 6, 12, 29, 10, 31], blue: 10 },
  { issue: "1024038", reds: [8, 10, 18, 23, 27, 31], blue: 2 },
  { issue: "1024037", reds: [1, 4, 5, 6, 12, 14], blue: 13 },
  { issue: "1024036", reds: [2, 8, 9, 12, 21, 31], blue: 2 },
  { issue: "1024035", reds: [5, 7, 14, 17, 22, 32], blue: 6 },
  { issue: "1024034", reds: [2, 9, 12, 19, 21, 31], blue: 4 },
  { issue: "1024033", reds: [6, 10, 11, 18, 10, 32], blue: 5 },
  { issue: "1024032", reds: [1, 3, 4, 11, 12, 21], blue: 16 },
];

export default function Home() {
  const [model, setModel] = useState<any>(null);
  const [prediction, setPrediction] = useState<number[] | null>(null);

  useEffect(() => {
    const initializeModel = async () => {
      if (typeof window !== "undefined" && window.ml5) {
        // 确保 TensorFlow.js 已经初始化
        await tf.ready();

        // 创建 ml5.js 模型
        const nn = window.ml5.neuralNetwork({
          task: "regression",
          debug: true,
        });

        setModel(nn);
      }
    };

    initializeModel();
  }, []);

  useEffect(() => {
    const trainAndPredict = async () => {
      if (model) {
        // 添加数据
        for (let i = 0; i < historyData.length - 10; i++) {
          const inputs: number[] = [];
          for (let j = i; j < i + 10; j++) {
            inputs.push(...historyData[j].reds, historyData[j].blue);
          }
          const outputs = [
            ...historyData[i + 10].reds,
            historyData[i + 10].blue,
          ];
          model.addData(inputs, outputs);
        }

        // 调试信息
        console.log("Added data to model:", model.data);

        // 训练模型
        model.normalizeData();
        await model.train({ epochs: 100 });

        console.log("Model trained!");

        // 进行预测
        const last10Entries = historyData.slice(-10);
        const inputData = last10Entries.flatMap((entry) => [
          ...entry.reds,
          entry.blue,
        ]);

        model.predict(inputData, (err: any, results: any) => {
          if (err) {
            console.error(err);
          } else {
            console.log("Raw prediction results:", results);

            let predictedNumbers;
            if (Array.isArray(results)) {
              predictedNumbers = results.map((r: any) => Math.round(r.value));
            } else if (typeof results === "object") {
              predictedNumbers = Object.values(results).map((value: any) =>
                Math.round(value)
              );
            } else {
              console.error("Unexpected results format:", results);
              return;
            }

            if (predictedNumbers.length === 7) {
              const adjustedPrediction = predictedNumbers.map((num, index) => {
                if (index < 6) {
                  return Math.max(1, Math.min(33, num));
                } else {
                  return Math.max(1, Math.min(16, num));
                }
              });

              setPrediction(adjustedPrediction);
            } else {
              console.error(
                "Prediction did not return 7 numbers:",
                predictedNumbers
              );
            }
          }
        });
      }
    };

    trainAndPredict();
  }, [model]);

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
      </div>
    </main>
  );
}
