import "@testing-library/jest-dom/vitest";

// AskPage 用 performance.now() 记「首个可见增量」耗时；jsdom 有 performance，
// 但 fetch/ReadableStream 在测试里由用例自己 stub，这里不做全局垫片——
// 默默补一个假 fetch 会让「测试通过」和「浏览器里能跑」再度分叉。
