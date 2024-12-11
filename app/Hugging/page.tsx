"use client";
import React, { useState, useEffect } from "react";
import { HfInference } from "@huggingface/inference";
import { historyData } from "@/app/data/thisData";
import dotenv from "dotenv";
dotenv.config();

const HuggingFace = () => {
  const [prediction, setPrediction] = useState<number[] | null>(null);

  const newhistoryData: Record<string, string[]> = {
    red1: historyData.map((item) => item.reds[0].toString()),
    red2: historyData.map((item) => item.reds[1].toString()),
    red3: historyData.map((item) => item.reds[2].toString()),
    red4: historyData.map((item) => item.reds[3].toString()),
    red5: historyData.map((item) => item.reds[4].toString()),
    red6: historyData.map((item) => item.reds[5].toString()),
    blue: historyData.map((item) => item.blue.toString()),
  };

  useEffect(() => {
    const predict = async () => {
      const hf = new HfInference(process.env.HuggingFace_token as string);
      console.log("New History Data:", newhistoryData);

      // Prepare input data
      const inputData = {
        data: newhistoryData,
      };

      const retry = async (retries: number) => {
        try {
          // Call the TabPFNMix regressor model for prediction
          const result = await hf.tabularRegression({
            model: "autogluon/tabpfn-mix-1.0-regressor",
            inputs: inputData,
          });

          setPrediction(result);
        } catch (error) {
          if (retries > 0) {
            console.error("Prediction error, retrying...", error);
            setTimeout(() => retry(retries - 1), 1000); // Retry after 1 second
          } else {
            console.error("Prediction error:", error);
          }
        }
      };

      retry(3); // Retry up to 3 times
    };

    predict();
  }, [newhistoryData]);

  return (
    <div>
      <h1>Prediction using TabPFNMix Regressor</h1>
      {prediction ? (
        <div>
          <h2>Predicted Values:</h2>
          <pre>{JSON.stringify(prediction, null, 2)}</pre>
        </div>
      ) : (
        <p>Loading prediction...</p>
      )}
    </div>
  );
};

export default HuggingFace;
