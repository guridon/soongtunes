import type { Metadata, Viewport } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "SoongTunes",
  description: "지식 그래프와 하이브리드 검색으로 대화하며 음악을 찾는 보드",
};

export const viewport: Viewport = { width: "device-width", initialScale: 1, themeColor: "#f3f2ef" };

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="ko">
      <body className="font-sans">{children}</body>
    </html>
  );
}
