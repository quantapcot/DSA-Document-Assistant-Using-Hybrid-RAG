import { createRequire } from "node:module";
import path from "node:path";
import { pathToFileURL } from "node:url";

const [runtime, output] = process.argv.slice(2);
const resolveArtifact = createRequire(path.join(runtime, "annotation.cjs"));
const { SpreadsheetFile, Workbook } = await import(
  pathToFileURL(resolveArtifact.resolve("@oai/artifact-tool")).href
);
const chunks = [];
for await (const chunk of process.stdin) chunks.push(chunk);
const rows = JSON.parse(Buffer.concat(chunks).toString("utf8"));

const workbook = Workbook.create();
const sheet = workbook.worksheets.add("Annotation");
sheet.showGridLines = false;
sheet.freezePanes.freezeRows(1);
const last = rows.length;
const grid = sheet.getRange(`A1:H${last}`);
grid.format.font = { name: "Arial", size: 10, color: "#16263D" };
grid.format.verticalAlignment = "top";

const widths = [85, 260, 245, 220, 62, 560, 130, 240];
for (const [index, width] of widths.entries()) {
  sheet.getRangeByIndexes(0, index, 1, 1).format.columnWidthPx = width;
}
for (const column of ["A", "B", "C", "D", "F", "H"]) {
  sheet.getRange(`${column}2:${column}${last}`).setNumberFormat("@");
}
// Source values belong in typed text cells, never in formulas.
grid.values = rows.map(row => row.map(value =>
  typeof value === "string" && value.startsWith("=") ? "'" + value : value
));
sheet.getRange(`B2:B${last}`).format.wrapText = true;
sheet.getRange(`C2:C${last}`).format.wrapText = true;
sheet.getRange(`D2:D${last}`).format.wrapText = true;
sheet.getRange(`F2:F${last}`).format.wrapText = true;
sheet.getRange(`H2:H${last}`).format.wrapText = true;
sheet.getRange(`G2:H${last}`).format.fill = "#FFFBEB";
sheet.getRange(`G2:G${last}`).dataValidation = {
  rule: { type: "list", values: ["0", "1", "2"] },
};
grid.format.autofitRows();

const header = sheet.getRange("A1:H1");
header.format.fill = "#203653";
header.format.font = { name: "Arial", size: 10, bold: true, color: "#FFFFFF" };
header.format.horizontalAlignment = "center";
header.format.verticalAlignment = "center";
header.format.rowHeight = 25;
workbook.recalculate();
const xlsx = await SpreadsheetFile.exportXlsx(workbook);
await xlsx.save(output);
