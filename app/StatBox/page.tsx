"use client";
import { Input } from "@/components/ui/input";
import React, { useState } from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Component } from "./chart";
import { Button } from "@/components/ui/button";
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

interface BallData {
  ball: number;
  frequent: number;
}

const StatBoxPage = () => {
  const [inputs, setInputs] = useState<string[]>([""]);
  const [redBalls, setRedBalls] = useState<number[]>([]);
  const [blueBalls, setBlueBalls] = useState<number[]>([]);

  const addInput = () => {
    setInputs([...inputs, ""]);
    console.log(inputs);
  };

  const handleChange = (index: number, value: string) => {
    console.log(index, value);
    const newInputs = [...inputs];
    newInputs[index] = value;
    setInputs(newInputs);

    if (value.trim()) {
      const numbers = value.split(" - ").map(Number);
      if (numbers.length === 7) {
        const newReds = numbers.slice(0, 6);
        const newBlue = numbers[6];

        // Append new values to the existing arrays
        setRedBalls((prevReds) => [...prevReds, ...newReds]);
        setBlueBalls((prevBlues) => [...prevBlues, newBlue]);
        console.log(redBalls, blueBalls);
      }
    } else {
      // If input is empty, remove corresponding index values
      const startRedIndex = index * 6;
      const endRedIndex = startRedIndex + 6;

      setRedBalls((prevReds) => [
        ...prevReds.slice(0, startRedIndex),
        ...prevReds.slice(endRedIndex),
      ]);

      setBlueBalls((prevBlues) => [
        ...prevBlues.slice(0, index),
        ...prevBlues.slice(index + 1),
      ]);

      console.log("🤔", redBalls, blueBalls);
    }
  };

  // Process ball frequency
  const processFrequency = (balls: number[]): BallData[] => {
    const frequencyMap: { [key: number]: number } = {};
    balls.forEach((ball) => {
      frequencyMap[ball] = (frequencyMap[ball] || 0) + 1;
    });

    return Object.entries(frequencyMap)
      .map(([ball, freq]) => ({
        ball: Number(ball),
        frequent: freq,
      }))
      .sort((a, b) => b.frequent - a.frequent);
  };

  // Get top N frequent balls
  const getTopFrequent = (balls: number[], topN: number): BallData[] => {
    return processFrequency(balls).slice(0, topN);
  };

  // Get red and blue ball statistics
  const topRedBalls = getTopFrequent(redBalls, 10);
  const topBlueBalls = getTopFrequent(blueBalls, 5);

  // Convert data format for chart component
  const redChartData = topRedBalls.map((item) => ({
    red: item.ball,
    Frequent: item.frequent,
  }));

  const blueChartData = topBlueBalls.map((item) => ({
    red: item.ball,
    Frequent: item.frequent,
  }));

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
      <Card className="container mx-auto my-36 w-5/6">
        <CardHeader>
          <CardTitle>Distribution</CardTitle>
          <CardDescription>Mode</CardDescription>
        </CardHeader>
        <CardContent className="flex flex-col gap-2">
          {inputs.map((input, index) => (
            <Input
              key={index}
              value={input}
              onChange={(e) => handleChange(index, e.target.value)}
              placeholder="Format: num - num - num - num - num - num - num"
            />
          ))}
          <Button onClick={addInput}>Add Input</Button>

          {/* Red balls chart */}
          {redChartData.length > 0 && (
            <div>
              <h3 className="mb-2">Top 10 Red Balls Chart</h3>
              <Component data={redChartData} />
            </div>
          )}

          {/* Blue balls chart */}
          {blueChartData.length > 0 && (
            <div className="mt-4">
              <h3 className="mb-2">Top 5 Blue Balls Chart</h3>
              <Component data={blueChartData} />
            </div>
          )}
        </CardContent>
        <CardFooter className="flex flex-col gap-2">
          <h3>
            Top 10 Red Balls:{" "}
            {topRedBalls
              .map((ball) => `${ball.ball} (${ball.frequent})`)
              .join(", ")}
          </h3>
          <h3>
            Top 5 Blue Balls:{" "}
            {topBlueBalls
              .map((ball) => `${ball.ball} (${ball.frequent})`)
              .join(", ")}
          </h3>
          <div className="text-sm text-gray-500"></div>
          Total red balls: {redBalls.length}, Total blue balls:{" "}
          {blueBalls.length}
        </CardFooter>
      </Card>
    </>
  );
};

export default StatBoxPage;
