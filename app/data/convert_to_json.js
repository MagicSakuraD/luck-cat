const fs = require("fs");
const path = require("path");

// 读取 TypeScript 文件
const tsFilePath = path.join(__dirname, "historyData.ts");
const tsContent = fs.readFileSync(tsFilePath, "utf-8");

// 使用正则表达式提取 historyData
const match = tsContent.match(/export const historyData = (\[.*?\]);/s);
if (!match) {
  throw new Error("historyData not found in the TypeScript file.");
}

const historyDataStr = match[1];
const historyData = eval(historyDataStr); // 注意：使用 eval 时要确保输入是可信的

// 写入 JSON 文件
const jsonFilePath = path.join(__dirname, "data.json");
fs.writeFileSync(jsonFilePath, JSON.stringify(historyData, null, 2));

console.log("Data has been converted to JSON and saved to data.json");
