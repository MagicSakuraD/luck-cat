"use client";
import { useEffect, useState } from "react";
// 移除不需要的TensorFlow导入
import { Component } from "../show";
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

// 不再需要PredictionResult接口，因为我们不使用ml5模型
interface StatOption {
  number: number;
  score: number;
}

// 常量定义
const RED_MAX = 33; // 红球最大号码
const BLUE_MAX = 16; // 蓝球最大号码
const TRAIN_NUMBER = 12; // 训练时使用的历史期数

// 定义统计模型配置 - 每个模型使用不同的参数组合
const modelConfigurations = [
  // 模型1: 历史频率优先
  {
    name: "历史频率",
    weights: {
      frequency: 0.6, // 历史频率权重
      recent: 0.2, // 最近号码权重
      uniqueness: 0.1, // 多样性权重
      random: 0.1, // 随机因素权重
    },
  },
  // 模型2: 最近趋势优先
  {
    name: "最近趋势",
    weights: {
      frequency: 0.3,
      recent: 0.5,
      uniqueness: 0.1,
      random: 0.1,
    },
  },
  // 模型3: 平衡策略
  {
    name: "平衡策略",
    weights: {
      frequency: 0.4,
      recent: 0.3,
      uniqueness: 0.2,
      random: 0.1,
    },
  },
];

export default function Home() {
  const [modelPredictions, setModelPredictions] = useState<number[][]>([]);
  const [aggregatedPrediction, setAggregatedPrediction] = useState<
    number[] | null
  >(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [currentStep, setCurrentStep] = useState<string>("");
  const [progress, setProgress] = useState<number>(0);
  const [show, setShow] = useState(true);
  const { historyData, isLoading, error } = useLotteryData();

  // 准备训练数据
  const prepareTrainingData = (position: number, isBlue = false) => {
    if (!historyData || historyData.length < TRAIN_NUMBER + 1) return null;

    const trainingData = [];

    // 从历史数据末尾开始，向前遍历
    for (let i = historyData.length - 1; i >= TRAIN_NUMBER; i--) {
      // 输入特征：过去12期的号码
      let inputs = [];
      for (let j = i; j > i - TRAIN_NUMBER; j--) {
        if (j < 0 || j >= historyData.length) continue;
        let entry = historyData[j];
        if (!entry || !entry.reds || !entry.blue) continue;
        inputs.push(...entry.reds, entry.blue);
      }

      // 输出目标：下一期的号码
      const nextEntryIndex = i - TRAIN_NUMBER;
      if (nextEntryIndex < 0 || nextEntryIndex >= historyData.length) continue;

      const nextEntry = historyData[nextEntryIndex];
      if (!nextEntry || !nextEntry.reds || nextEntry.blue === undefined)
        continue;

      // 安全地访问数据
      const target = isBlue
        ? nextEntry.blue.toString() // 蓝球（作为字符串标签）
        : (nextEntry.reds[position] || position + 1).toString(); // 特定位置的红球，如果没有则使用默认值

      trainingData.push({ inputs, output: target });
    }

    return trainingData;
  };

  // 统计预测函数 - 使用模型配置中的权重参数
  const predictWithStats = async (
    position: number,
    isBlue: boolean,
    modelIndex: number
  ): Promise<number> => {
    console.log(
      `开始统计预测 ${isBlue ? "蓝球" : "红球" + position} (模型${
        modelIndex + 1
      })`
    );

    try {
      // 使用训练数据集
      const trainingData = prepareTrainingData(position, isBlue);

      if (!trainingData || trainingData.length === 0) {
        console.log(`没有训练数据，使用默认值`);
        return isBlue ? 1 : position + 1;
      }

      // 获取当前模型的权重配置
      const weights = modelConfigurations[modelIndex].weights;
      const maxNum = isBlue ? BLUE_MAX : RED_MAX;

      // 分析历史数据的频率分布
      const frequency: Record<string, number> = {};
      const recent: Record<string, number> = {}; // 最近几期的权重更高

      // 为所有可能的号码设置基础频率
      for (let i = 1; i <= maxNum; i++) {
        frequency[i.toString()] = 0;
        recent[i.toString()] = 0;
      }

      // 计算历史频率
      trainingData.forEach((item, index) => {
        if (!item.output) return;
        frequency[item.output] = (frequency[item.output] || 0) + 1;

        // 最近5期的号码权重更高
        if (index < 5) {
          recent[item.output] = (recent[item.output] || 0) + (5 - index); // 越近权重越高
        }
      });

      // 计算一致性和多样性指标
      const consistencyScore: Record<string, number> = {};
      const uniquenessScore: Record<string, number> = {};

      // 检查历史数据中号码的分布模式
      for (let i = 1; i <= maxNum; i++) {
        const num = i.toString();
        // 一致性基于历史频率
        consistencyScore[num] = (frequency[num] || 0) / trainingData.length;

        // 多样性指标 - 如果这个号码很少出现，给它一个机会
        uniquenessScore[num] = 1 - consistencyScore[num];
      }

      // 综合评分 (使用模型配置的权重)
      const finalScores: Record<string, number> = {};

      for (let i = 1; i <= maxNum; i++) {
        const num = i.toString();
        // 使用模型配置的权重计算综合得分
        const recentWeight = recent[num] ? recent[num] / 15 : 0; // 归一化，最高15分

        finalScores[num] =
          weights.frequency * consistencyScore[num] +
          weights.recent * recentWeight +
          weights.uniqueness * uniquenessScore[num] +
          weights.random * Math.random();
      }

      // 将评分转换为排序列表
      const sortedNumbers = Object.entries(finalScores)
        .map(([num, score]) => ({ number: parseInt(num), score }))
        .sort((a, b) => b.score - a.score);

      // 选择策略
      // 85%的时间选择前5名中的一个，15%的时间随机选择
      if (Math.random() < 0.85 && sortedNumbers.length >= 5) {
        // 加权随机选择前5名
        const top5 = sortedNumbers.slice(0, 5);
        const totalWeight = top5.reduce((sum, item) => sum + item.score, 0);
        let randomWeight = Math.random() * totalWeight;

        for (const item of top5) {
          randomWeight -= item.score;
          if (randomWeight <= 0) {
            return item.number;
          }
        }
        return top5[0].number; // 防止浮点误差
      } else {
        // 完全随机选择
        return Math.floor(Math.random() * maxNum) + 1;
      }
    } catch (error) {
      console.error(`预测错误:`, error);
      return isBlue ? 1 : position + 1;
    }
  };

  // 运行所有模型的预测
  const runAllModels = async () => {
    setShow(false);
    setLoading(true);
    setModelPredictions([]);
    setProgress(0);

    try {
      const allPredictions: number[][] = [];

      // 对每个模型配置进行预测
      for (
        let modelIndex = 0;
        modelIndex < modelConfigurations.length;
        modelIndex++
      ) {
        try {
          setCurrentStep(
            `计算模型 ${modelIndex + 1}/${modelConfigurations.length}`
          );

          const modelPredictions: number[] = [];

          // 预测红球
          for (let position = 0; position < 6; position++) {
            setCurrentStep(
              `预测红球 ${position + 1}/6 (模型 ${modelIndex + 1})`
            );

            try {
              const prediction = await predictWithStats(
                position,
                false,
                modelIndex
              );
              modelPredictions.push(prediction);
            } catch (err) {
              console.error(`红球${position + 1}预测失败:`, err);
              // 使用默认值
              modelPredictions.push(position + 1);
            }

            // 更新进度
            setProgress(
              ((position + modelIndex * 7) / (modelConfigurations.length * 7)) *
                100
            );
          }

          // 预测蓝球
          setCurrentStep(`预测蓝球 (模型 ${modelIndex + 1})`);
          try {
            const bluePrediction = await predictWithStats(0, true, modelIndex);
            modelPredictions.push(bluePrediction);
          } catch (err) {
            console.error("蓝球预测失败:", err);
            modelPredictions.push(1); // 默认值
          }

          // 添加到所有预测
          allPredictions.push(modelPredictions);
          setModelPredictions([...allPredictions]);

          // 更新进度
          setProgress(((modelIndex + 1) / modelConfigurations.length) * 100);
        } catch (modelError) {
          console.error(`模型 ${modelIndex + 1} 处理失败:`, modelError);
          // 继续下一个模型
          continue;
        }
      }

      // 所有模型完成后，计算聚合预测
      if (allPredictions.length > 0) {
        const aggregated = aggregatePredictions(allPredictions);
        setAggregatedPrediction(aggregated);
      }
    } catch (error) {
      console.error("预测过程中发生错误:", error);
    } finally {
      setLoading(false);
    }
  };

  // 聚合预测结果
  const aggregatePredictions = (predictions: number[][]): number[] => {
    // 红球票选结果
    const redResults = [];

    for (let pos = 0; pos < 6; pos++) {
      const counters: Record<number, number> = {};

      // 统计每个号码在该位置的出现次数
      predictions.forEach((pred) => {
        if (!pred || pred.length <= pos) return;
        const val = pred[pos];
        counters[val] = (counters[val] || 0) + 1;
      });

      // 找出出现最多次的号码
      let maxCount = 0;
      let mostFrequent = pos + 1; // 默认值

      Object.entries(counters).forEach(([numStr, count]) => {
        const num = parseInt(numStr);
        if (count > maxCount) {
          maxCount = count;
          mostFrequent = num;
        }
      });

      redResults.push(mostFrequent);
    }

    // 确保红球不重复并排序
    const uniqueReds = Array.from(new Set(redResults));
    while (uniqueReds.length < 6) {
      const randomNum = Math.floor(Math.random() * RED_MAX) + 1;
      if (!uniqueReds.includes(randomNum)) {
        uniqueReds.push(randomNum);
      }
    }
    const sortedReds = [...uniqueReds].sort((a, b) => a - b);

    // 蓝球票选结果
    const blueCounters: Record<number, number> = {};
    predictions.forEach((pred) => {
      if (!pred || pred.length < 7) return;
      const blue = pred[6];
      blueCounters[blue] = (blueCounters[blue] || 0) + 1;
    });

    let maxBlueCount = 0;
    let mostFrequentBlue = 1; // 默认值

    Object.entries(blueCounters).forEach(([numStr, count]) => {
      const num = parseInt(numStr);
      if (count > maxBlueCount) {
        maxBlueCount = count;
        mostFrequentBlue = num;
      }
    });

    // 返回完整预测结果
    return [...sortedReds.slice(0, 6), mostFrequentBlue];
  };

  // 加载状态展示
  if (isLoading) {
    return (
      <div className="text-center w-full mt-20 text-blue-600 font-bold text-lg">
        加载数据中...
      </div>
    );
  }

  // 错误状态展示
  if (error) {
    return (
      <div className="text-center w-full mt-20 text-rose-600 font-bold text-lg">
        错误: {error}
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
              <BreadcrumbLink href="#">预测</BreadcrumbLink>
            </BreadcrumbItem>
            <BreadcrumbSeparator className="hidden md:block" />
            <BreadcrumbItem>
              <BreadcrumbPage>统计方法预测</BreadcrumbPage>
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
                  {currentStep} ({Math.round(progress)}%)
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
                    最终预测结果 (统计方法)
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
                        <h4 className="font-medium mb-2">
                          {modelConfigurations[idx].name}
                        </h4>
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
              disabled={loading}
              className="bg-blue-600 hover:bg-blue-700 text-white"
            >
              开始预测 (统计方法)
            </Button>
          </CardFooter>
        )}
      </div>
    </>
  );
}
