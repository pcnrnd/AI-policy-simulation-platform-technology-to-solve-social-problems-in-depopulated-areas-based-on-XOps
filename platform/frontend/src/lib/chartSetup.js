// Chart.js 전역 등록 — 앱 진입 시 1회. 8종 시각화에 필요한 컨트롤러/요소를 모두 등록한다.
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  RadialLinearScale,
  BarElement,
  PointElement,
  LineElement,
  ArcElement,
  Title,
  Tooltip,
  Legend,
  Filler
} from "chart.js";

ChartJS.register(
  CategoryScale,
  LinearScale,
  RadialLinearScale,
  BarElement,
  PointElement,
  LineElement,
  ArcElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

// CSS media query가 제어하지 못하는 canvas 애니메이션도 운영체제 모션 축소 설정을 따른다.
if (typeof window !== "undefined" && window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
  ChartJS.defaults.animation = false;
}

export default ChartJS;
