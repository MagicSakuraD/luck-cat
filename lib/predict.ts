// // 选择包含数据的表格行
// const rows = document.querySelectorAll("#tdata .t_tr1");

// // 存储提取的数据
// const data = [];

// // 遍历表格行
// rows.forEach((row) => {
//   const cells = row.querySelectorAll("td");

//   // 提取期号
//   const issue = cells[0]?.innerText.trim();

//   // 提取红球号码
//   const reds = [];
//   for (let j = 1; j <= 6; j++) {
//     const redNumber = parseInt(cells[j]?.innerText.trim(), 10);
//     if (!isNaN(redNumber)) {
//       reds.push(redNumber);
//     }
//   }

//   // 提取蓝球号码
//   const blue = parseInt(cells[7]?.innerText.trim(), 10);

//   // 添加到数据数组
//   if (issue && reds.length === 6 && !isNaN(blue)) {
//     data.push({ issue, reds, blue });
//   }
// });

// // 格式化输出数据
// const formattedData = data.map((item) => JSON.stringify(item)).join(",\n");

// // 打印格式化的数据到控制台
// console.log("[\n" + formattedData + "\n]");
