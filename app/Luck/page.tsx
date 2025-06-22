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

// Enhanced statistical model configurations with optimized weights
//
// 关于本福特定律 (Benford's Law) 偏向的说明：
// 本福特定律表明在许多自然数据集中，小数字（如1,2,3）作为首位数字出现的频率
// 比大数字更高。虽然理论上彩票号码是均匀分布的，但我们添加这个偏向作为一种
// "非理性扰动"，用于实验不同的预测策略。这个偏向权重通常很小（0.01-0.02），
// 仅作为多元化预测策略的一部分。
//
const modelConfigurations = [
  // 模型1: 频率与趋势平衡
  {
    name: "频率趋势平衡",
    weights: {
      frequency: 0.45,
      recent: 0.35,
      uniqueness: 0.15,
      random: 0.05,
      benfordBias: 0.01, // 本福特偏向（非理性扰动）
    },
  },
  // 模型2: 最近热号优先
  {
    name: "热号追踪",
    weights: {
      frequency: 0.25,
      recent: 0.55,
      uniqueness: 0.1,
      random: 0.1,
      benfordBias: 0.0, // 不需要此扰动
    },
  },
  // 模型3: 冷号回补策略
  {
    name: "冷号回补",
    weights: {
      frequency: 0.2,
      recent: 0.2,
      uniqueness: 0.5,
      random: 0.1,
      benfordBias: 0.01, // 轻微扰动
    },
  },
  // 模型4: 均衡策略
  {
    name: "均衡策略",
    weights: {
      frequency: 0.35,
      recent: 0.25,
      uniqueness: 0.25,
      random: 0.15,
      benfordBias: 0.0,
    },
  },
  // 模型5: 历史主导
  {
    name: "历史主导",
    weights: {
      frequency: 0.6,
      recent: 0.2,
      uniqueness: 0.1,
      random: 0.1,
      benfordBias: 0.0,
    },
  },
  // 模型6: 随机融合
  {
    name: "随机融合",
    weights: {
      frequency: 0.3,
      recent: 0.3,
      uniqueness: 0.2,
      random: 0.2,
      benfordBias: 0.02, // 较高扰动
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
  // 统计预测函数 - 增强版，加入更多统计学特征
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
      const trainingData = prepareTrainingData(position, isBlue);
      if (!trainingData || trainingData.length === 0) {
        console.log(`没有训练数据，使用默认值`);
        return isBlue ? 1 : position + 1;
      }

      const weights = modelConfigurations[modelIndex].weights;
      const maxNum = isBlue ? BLUE_MAX : RED_MAX;

      // 基础频率分析
      const frequency: Record<string, number> = {};
      const recent: Record<string, number> = {};

      // 新增统计学特征
      const intervalAnalysis: Record<string, number> = {}; // 间隔分析
      const positionBias: Record<string, number> = {}; // 位置偏向
      const cyclicPattern: Record<string, number> = {}; // 周期性模式

      // 初始化
      for (let i = 1; i <= maxNum; i++) {
        const key = i.toString();
        frequency[key] = 0;
        recent[key] = 0;
        intervalAnalysis[key] = 0;
        positionBias[key] = 0;
        cyclicPattern[key] = 0;
      }

      // 分析历史数据
      trainingData.forEach((item, index) => {
        if (!item.output) return;
        const num = item.output;

        // 基础频率
        frequency[num] = (frequency[num] || 0) + 1;

        // 最近趋势权重
        if (index < 5) {
          recent[num] = (recent[num] || 0) + (5 - index);
        }

        // 间隔分析 - 计算号码出现的间隔模式
        const lastAppearance = trainingData.findIndex(
          (data, i) => i > index && data.output === num
        );
        if (lastAppearance !== -1) {
          const interval = lastAppearance - index;
          intervalAnalysis[num] += 1 / Math.max(interval, 1); // 间隔越小权重越高
        }

        // 周期性模式分析 - 检查是否有7天、14天等周期
        const weekPattern = index % 7;
        const biWeekPattern = index % 14;
        cyclicPattern[num] +=
          (weekPattern === 0 ? 0.1 : 0) + (biWeekPattern === 0 ? 0.05 : 0);
      });

      // 计算综合评分
      const finalScores: Record<string, number> = {};

      for (let i = 1; i <= maxNum; i++) {
        const num = i.toString();

        // 标准化各项指标
        const freqScore = (frequency[num] || 0) / trainingData.length;
        const recentScore = (recent[num] || 0) / 15;
        const intervalScore = intervalAnalysis[num] || 0;
        const cyclicScore = cyclicPattern[num] || 0;

        // 多样性指标 - 避免过度集中
        const uniquenessScore = 1 - freqScore; // 数字和谐性 - 相邻数字的关联性（轻微的偏向）
        const neighborEffect = isBlue
          ? 0
          : (((frequency[(i - 1).toString()] || 0) +
              (frequency[(i + 1).toString()] || 0)) /
              (trainingData.length * 2)) *
            0.1;

        // 本福特定律偏向 - 小数字有更高的概率（非理性扰动）
        // 根据本福特定律，数字1的概率约为30.1%，随数字增大而递减
        const benfordScore = Math.log10(1 + 1 / i); // 本福特定律公式
        const normalizedBenfordScore = benfordScore / Math.log10(2); // 标准化到0-1范围

        // 综合评分（加权求和）
        finalScores[num] =
          weights.frequency * freqScore +
          weights.recent * recentScore +
          weights.uniqueness * uniquenessScore +
          0.1 * intervalScore +
          0.05 * cyclicScore +
          0.05 * neighborEffect +
          weights.benfordBias * normalizedBenfordScore +
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
          } // 预测蓝球
          setCurrentStep(`预测蓝球 (模型 ${modelIndex + 1})`);
          try {
            const bluePrediction = await predictWithStats(0, true, modelIndex);
            modelPredictions.push(bluePrediction);
          } catch (err) {
            console.error("蓝球预测失败:", err);
            modelPredictions.push(1); // 默认值
          }

          // 应用红球去重和排序优化 🎯
          const optimizedPrediction =
            ensureUniqueAndSortedRedBalls(modelPredictions);
          console.log(
            `模型 ${modelIndex + 1} (${
              modelConfigurations[modelIndex].name
            }) 去重排序后:`,
            optimizedPrediction
          );

          // 添加到所有预测
          allPredictions.push(optimizedPrediction);
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
  // 聚合预测结果 - 改进版，防止重复
  const aggregatePredictions = (predictions: number[][]): number[] => {
    // 统计所有红球号码的总体频率
    const globalRedFreq: Record<number, number> = {};
    for (let i = 1; i <= RED_MAX; i++) {
      globalRedFreq[i] = 0;
    }

    // 统计每个模型预测的红球频率
    predictions.forEach((pred) => {
      if (!pred || pred.length < 6) return;
      for (let i = 0; i < 6; i++) {
        const num = pred[i];
        if (num >= 1 && num <= RED_MAX) {
          globalRedFreq[num] = (globalRedFreq[num] || 0) + 1;
        }
      }
    });

    // 按频率排序，选择频率最高的6个不重复号码
    const sortedByFreq = Object.entries(globalRedFreq)
      .map(([numStr, freq]) => ({ num: parseInt(numStr), freq }))
      .sort((a, b) => {
        if (b.freq !== a.freq) return b.freq - a.freq; // 频率优先
        return Math.random() - 0.5; // 频率相同时随机排序
      });

    // 选择前6个号码作为红球
    const selectedReds = sortedByFreq.slice(0, 6).map((item) => item.num);

    // 如果频率都相同（极少情况），补充随机号码
    while (selectedReds.length < 6) {
      const randomNum = Math.floor(Math.random() * RED_MAX) + 1;
      if (!selectedReds.includes(randomNum)) {
        selectedReds.push(randomNum);
      }
    }

    const sortedReds = selectedReds.sort((a, b) => a - b);

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
    }); // 返回完整预测结果，确保红球已排序
    const finalResult = [...sortedReds.slice(0, 6), mostFrequentBlue];
    console.log("聚合预测结果 (已排序):", finalResult);
    return finalResult;
  };

  // 红球去重和排序优化函数
  const ensureUniqueAndSortedRedBalls = (prediction: number[]): number[] => {
    const redBalls = prediction.slice(0, 6);
    const blueBall = prediction[6];

    // 去重红球
    const uniqueReds = Array.from(
      new Set(redBalls.filter((num) => num >= 1 && num <= RED_MAX))
    );

    if (uniqueReds.length === 6) {
      // 没有重复且数量正确，直接排序返回
      return [...uniqueReds.sort((a, b) => a - b), blueBall];
    }

    // 需要补充号码
    const usedNumbers = new Set(uniqueReds);
    const finalReds = [...uniqueReds];

    // 为缺少的球位选择新号码，使用统计学策略
    while (finalReds.length < 6) {
      let newNumber;

      // 30% 概率选择统计学上的"温和数字" (中位数附近)
      if (Math.random() < 0.3) {
        const mildNumbers = [14, 15, 16, 17, 18, 19, 20].filter(
          (n) => !usedNumbers.has(n)
        );
        if (mildNumbers.length > 0) {
          newNumber =
            mildNumbers[Math.floor(Math.random() * mildNumbers.length)];
        }
      }

      // 25% 概率选择"边缘数字" (1-10, 24-33)
      if (!newNumber && Math.random() < 0.25) {
        const edgeNumbers = [];
        for (let i = 1; i <= 10; i++) {
          if (!usedNumbers.has(i)) edgeNumbers.push(i);
        }
        for (let i = 24; i <= 33; i++) {
          if (!usedNumbers.has(i)) edgeNumbers.push(i);
        }
        if (edgeNumbers.length > 0) {
          newNumber =
            edgeNumbers[Math.floor(Math.random() * edgeNumbers.length)];
        }
      }

      // 其余情况：均匀随机选择
      if (!newNumber) {
        const candidates = [];
        for (let i = 1; i <= RED_MAX; i++) {
          if (!usedNumbers.has(i)) {
            candidates.push(i);
          }
        }

        if (candidates.length > 0) {
          newNumber = candidates[Math.floor(Math.random() * candidates.length)];
        } else {
          // 最后兜底：直接赋值
          for (let i = 1; i <= RED_MAX; i++) {
            if (!usedNumbers.has(i)) {
              newNumber = i;
              break;
            }
          }
        }
      }

      if (newNumber) {
        finalReds.push(newNumber);
        usedNumbers.add(newNumber);
      }
    }

    // 返回排序后的红球 + 蓝球
    return [...finalReds.sort((a, b) => a - b), blueBall];
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
              {" "}
              {aggregatedPrediction && (
                <div className="mb-8">
                  <h2 className="text-xl font-bold mb-4 text-center">
                    最终预测结果 (统计方法)
                  </h2>{" "}
                  <p className="text-sm text-gray-600 text-center mb-4">
                    基于多模型聚合，包含频率分析、趋势预测、本福特定律等多种统计学方法
                    <br />
                    <span className="text-green-600 font-medium">
                      ✓ 红球自动去重排序 (1-33)
                    </span>{" "}
                    |
                    <span className="text-blue-600 font-medium">
                      蓝球范围 (1-16)
                    </span>
                  </p>
                  <Component visitors={aggregatedPrediction} />
                </div>
              )}
              {modelPredictions.length > 0 && (
                <div>
                  {" "}
                  <h3 className="text-lg font-bold mt-8 mb-4">
                    各模型预测结果{" "}
                    <span className="text-sm font-normal text-gray-500">
                      (红球已排序去重)
                    </span>
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
