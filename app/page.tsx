"use client";
import { useEffect, useState } from "react";
import * as tf from "@tensorflow/tfjs";
import { LeafyGreen } from "lucide-react";

// 假设这是最近期的数据
const historyData = [
  // { issue: "2024086", reds: [17, 19, 20, 23, 25, 31], blue: 2 },
  { issue: "2024085", reds: [1, 5, 15, 21, 23, 27], blue: 15 },
  { issue: "2024084", reds: [1, 8, 10, 13, 19, 29], blue: 13 },
  { issue: "2024083", reds: [3, 9, 14, 29, 32, 33], blue: 15 },
  { issue: "2024082", reds: [1, 2, 13, 16, 17, 29], blue: 16 },
  { issue: "2024081", reds: [1, 6, 12, 17, 23, 25], blue: 4 },
  { issue: "2024080", reds: [6, 11, 12, 27, 29, 30], blue: 13 },
  { issue: "2024079", reds: [2, 3, 6, 7, 16, 26], blue: 4 },
  { issue: "2024078", reds: [5, 9, 14, 21, 22, 26], blue: 12 },
  { issue: "2024077", reds: [1, 4, 6, 14, 17, 22], blue: 8 },
  { issue: "2024076", reds: [3, 22, 24, 27, 29, 32], blue: 15 },
  { issue: "2024075", reds: [3, 5, 8, 18, 22, 28], blue: 1 },
  { issue: "2024074", reds: [7, 8, 10, 22, 24, 32], blue: 7 },
  { issue: "2024073", reds: [3, 9, 12, 18, 28, 30], blue: 1 },
  { issue: "2024072", reds: [1, 14, 20, 21, 23, 27], blue: 6 },
  { issue: "2024071", reds: [2, 8, 19, 28, 30, 31], blue: 14 },
  { issue: "2024070", reds: [4, 13, 18, 20, 22, 28], blue: 5 },
  { issue: "2024069", reds: [6, 13, 20, 21, 24, 32], blue: 6 },
  { issue: "2024068", reds: [1, 2, 7, 19, 20, 21], blue: 1 },
  { issue: "2024067", reds: [1, 11, 13, 17, 25, 29], blue: 3 },
  { issue: "2024066", reds: [8, 9, 12, 22, 26, 32], blue: 13 },
  { issue: "2024065", reds: [3, 5, 9, 10, 19, 22], blue: 14 },
  { issue: "2024064", reds: [6, 8, 17, 18, 28, 30], blue: 5 },
  { issue: "2024063", reds: [7, 14, 16, 23, 28, 32], blue: 4 },
  { issue: "2024062", reds: [1, 7, 10, 16, 18, 27], blue: 16 },
  { issue: "2024061", reds: [1, 9, 18, 22, 25, 28], blue: 2 },
  { issue: "2024060", reds: [1, 2, 6, 10, 22, 28], blue: 15 },
  { issue: "2024059", reds: [1, 3, 14, 25, 31, 33], blue: 7 },
  { issue: "2024058", reds: [8, 12, 13, 17, 27, 29], blue: 13 },
  { issue: "2024057", reds: [3, 8, 17, 18, 23, 31], blue: 8 },
  { issue: "2024056", reds: [5, 18, 20, 24, 25, 26], blue: 6 },
  { issue: "2024055", reds: [2, 4, 6, 7, 16, 29], blue: 3 },
  { issue: "2024054", reds: [8, 13, 20, 25, 31, 32], blue: 3 },
  { issue: "2024053", reds: [7, 14, 21, 22, 28, 33], blue: 7 },
  { issue: "2024052", reds: [7, 10, 11, 15, 17, 21], blue: 3 },
  { issue: "2024051", reds: [5, 9, 13, 20, 23, 28], blue: 6 },
  { issue: "2024050", reds: [1, 3, 7, 10, 22, 33], blue: 2 },
  { issue: "2024049", reds: [12, 15, 17, 23, 26, 32], blue: 11 },
  { issue: "2024048", reds: [2, 9, 15, 19, 26, 28], blue: 2 },
  { issue: "2024047", reds: [7, 8, 21, 26, 29, 30], blue: 15 },
  { issue: "2024046", reds: [2, 6, 10, 11, 17, 29], blue: 15 },
  { issue: "2024045", reds: [2, 8, 19, 23, 24, 26], blue: 3 },
  { issue: "2024044", reds: [2, 6, 17, 25, 32, 33], blue: 6 },
  { issue: "2024043", reds: [4, 6, 7, 14, 15, 24], blue: 8 },
  { issue: "2024042", reds: [2, 4, 5, 14, 26, 32], blue: 14 },
  { issue: "2024041", reds: [2, 9, 12, 22, 25, 33], blue: 16 },
  { issue: "2024040", reds: [11, 14, 18, 19, 23, 26], blue: 2 },
  { issue: "2024039", reds: [2, 6, 12, 29, 30, 31], blue: 10 },
  { issue: "2024038", reds: [8, 10, 18, 23, 27, 31], blue: 2 },
  { issue: "2024037", reds: [1, 4, 5, 6, 12, 14], blue: 13 },
  { issue: "2024036", reds: [2, 8, 9, 12, 21, 31], blue: 2 },
  { issue: "2024035", reds: [5, 7, 14, 17, 22, 32], blue: 6 },
  { issue: "2024034", reds: [2, 9, 12, 19, 21, 31], blue: 4 },
  { issue: "2024033", reds: [6, 10, 11, 18, 20, 32], blue: 5 },
  { issue: "2024032", reds: [1, 3, 4, 11, 12, 21], blue: 16 },

  {
    issue: "2024031",
    reds: [2, 5, 7, 13, 19, 26],
    blue: 8,
  },
  {
    issue: "2024030",
    reds: [3, 6, 9, 15, 22, 28],
    blue: 12,
  },
  {
    issue: "2024029",
    reds: [1, 4, 8, 16, 23, 30],
    blue: 6,
  },
  {
    issue: "2024028",
    reds: [2, 5, 10, 17, 24, 31],
    blue: 11,
  },
  {
    issue: "2024027",
    reds: [3, 6, 11, 18, 25, 32],
    blue: 5,
  },
  {
    issue: "2024026",
    reds: [1, 3, 7, 10, 22, 33],
    blue: 2,
  },
  {
    issue: "2024025",
    reds: [5, 9, 13, 20, 23, 28],
    blue: 6,
  },
  {
    issue: "2024024",
    reds: [7, 10, 11, 15, 17, 21],
    blue: 3,
  },
  {
    issue: "2024023",
    reds: [5, 7, 14, 17, 22, 32],
    blue: 6,
  },
  {
    issue: "2024022",
    reds: [2, 4, 6, 7, 16, 29],
    blue: 3,
  },
  {
    issue: "2024021",
    reds: [8, 13, 20, 25, 31, 32],
    blue: 3,
  },
  {
    issue: "2024020",
    reds: [7, 14, 21, 22, 28, 33],
    blue: 7,
  },
  {
    issue: "2024019",
    reds: [8, 12, 13, 17, 27, 29],
    blue: 13,
  },
  {
    issue: "2024018",
    reds: [3, 8, 17, 18, 23, 31],
    blue: 8,
  },
  {
    issue: "2024017",
    reds: [5, 18, 20, 24, 25, 26],
    blue: 6,
  },
  {
    issue: "2024016",
    reds: [2, 4, 6, 7, 16, 29],
    blue: 3,
  },
  {
    issue: "2024015",
    reds: [8, 13, 20, 25, 31, 32],
    blue: 3,
  },
  {
    issue: "2024014",
    reds: [7, 14, 21, 22, 28, 33],
    blue: 7,
  },
  {
    issue: "2024013",
    reds: [8, 12, 13, 17, 27, 29],
    blue: 13,
  },
  {
    issue: "2024012",
    reds: [3, 8, 17, 18, 23, 31],
    blue: 8,
  },
  {
    issue: "2024011",
    reds: [5, 18, 20, 24, 25, 26],
    blue: 6,
  },
  {
    issue: "2024010",
    reds: [2, 4, 6, 7, 16, 29],
    blue: 3,
  },
  {
    issue: "2024009",
    reds: [8, 13, 20, 25, 31, 32],
    blue: 3,
  },
  {
    issue: "2024008",
    reds: [7, 14, 21, 22, 28, 33],
    blue: 7,
  },
  {
    issue: "2024007",
    reds: [8, 12, 13, 17, 27, 29],
    blue: 13,
  },
  {
    issue: "2024006",
    reds: [3, 8, 17, 18, 23, 31],
    blue: 8,
  },
  {
    issue: "2024005",
    reds: [5, 18, 20, 24, 25, 26],
    blue: 6,
  },
  {
    issue: "2024004",
    reds: [2, 4, 6, 7, 16, 29],
    blue: 3,
  },
  {
    issue: "2024003",
    reds: [8, 13, 20, 25, 31, 32],
    blue: 3,
  },
  {
    issue: "2024002",
    reds: [7, 14, 21, 22, 28, 33],
    blue: 7,
  },
  {
    issue: "2024001",
    reds: [8, 12, 13, 17, 27, 29],
    blue: 13,
  },
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
          layers: [
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
              units: 7,
              activation: "sigmoid", // 使用sigmoid确保输出在0-1之间
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
        for (let i = historyData.length - 1; i >= 10; i--) {
          const inputs: number[] = [];
          for (let j = i; j > i - 10; j--) {
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
            ...historyData[i - 10].reds,
            historyData[i - 10].blue,
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
          learningRate: 0.001,
          callbacks: {
            onEpochEnd: async (epoch: number, logs: any) => {
              console.log(`Epoch ${epoch}: loss = ${logs.loss}`);
              if (logs.loss < 0.05) {
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
        const last10Entries = historyData.slice(0, 10);
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
