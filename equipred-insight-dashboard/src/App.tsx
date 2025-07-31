
import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import React, { Suspense, lazy } from "react";
import Index from "./pages/Index";
import NotFound from "./pages/NotFound";

// Lazy load pages for code splitting
const Visualizations = lazy(() => import("./pages/Visualizations"));
const Predictions = lazy(() => import("./pages/Predictions"));
const Dataset = lazy(() => import("./pages/Dataset"));
const Settings = lazy(() => import("./pages/Settings"));

// Loading fallback component
const LoadingFallback = () => (
  <div className="flex items-center justify-center h-screen">
    <div className="animate-spin rounded-full h-10 w-10 border-b-2 border-blue-600"></div>
    <span className="ml-3 text-lg">Loading...</span>
  </div>
);

const queryClient = new QueryClient();

const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <Toaster />
      <Sonner />
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Index />} />
          <Route path="/visualizations" element={
            <Suspense fallback={<LoadingFallback />}>
              <Visualizations />
            </Suspense>
          } />
          <Route path="/predictions" element={
            <Suspense fallback={<LoadingFallback />}>
              <Predictions />
            </Suspense>
          } />
          <Route path="/dataset" element={
            <Suspense fallback={<LoadingFallback />}>
              <Dataset />
            </Suspense>
          } />
          <Route path="/settings" element={
            <Suspense fallback={<LoadingFallback />}>
              <Settings />
            </Suspense>
          } />
          {/* ADD ALL CUSTOM ROUTES ABOVE THE CATCH-ALL "*" ROUTE */}
          <Route path="*" element={<NotFound />} />
        </Routes>
      </BrowserRouter>
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;
