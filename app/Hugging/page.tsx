"use client";
import React, { useState, useEffect } from "react";
import { HfInference } from "@huggingface/inference";
import { historyData } from "@/app/data/thisData";
import dotenv from "dotenv";
dotenv.config();

if (!process.env.NEXT_PUBLIC_HuggingFace_token) {
  throw new Error("Missing Hugging Face token in environment variables");
}

const HuggingFace = () => {
  const [prediction, setPrediction] = useState<number[] | null>(null);

  const newhistoryData = historyData.map((item) => ({
    red1: item.reds[0],
    red2: item.reds[1],
    red3: item.reds[2],
    red4: item.reds[3],
    red5: item.reds[4],
    red6: item.reds[5],
    blue: item.blue,
  }));

  useEffect(() => {
    const predict = async () => {
      const hf = new HfInference(
        process.env.NEXT_PUBLIC_HuggingFace_token as string
      );
      console.log("New History Data:", newhistoryData);

      // Prepare input data
      const inputData = {
        data: {
          red1: newhistoryData.map((item) => item.red1.toString()),
          red2: newhistoryData.map((item) => item.red2.toString()),
          red3: newhistoryData.map((item) => item.red3.toString()),
          red4: newhistoryData.map((item) => item.red4.toString()),
          red5: newhistoryData.map((item) => item.red5.toString()),
          red6: newhistoryData.map((item) => item.red6.toString()),
          blue: newhistoryData.map((item) => item.blue.toString()),
        },
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
