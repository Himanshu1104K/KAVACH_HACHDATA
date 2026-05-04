import NavBar from "../components/NavBar";
import EfficencyGraph from "../components/EfficencyGraph";
import StrikeGraph from "../components/StrikeGraph";
import StrikeList from "../components/StrikeList";
import PageTransition from "../components/PageTransition";

const MainDash = () => {
  return (
    <div className="ui-shell min-h-screen">
      <NavBar />
      <div className="mx-auto max-w-7xl px-4 pb-10 pt-16 sm:px-5 sm:pt-20">
        <PageTransition>
          <header className="mb-8 text-center sm:mb-10">
            <h1 className="mx-auto max-w-2xl text-2xl font-semibold tracking-tight sm:text-3xl md:text-4xl">
              <span className="bg-gradient-to-r from-white via-[#e8f2ff] to-emerald-200/85 bg-clip-text text-transparent">
                Operations dashboard
              </span>
            </h1>
            <div
              className="mx-auto mt-4 h-px w-20 bg-gradient-to-r from-transparent via-emerald-400/45 to-transparent sm:mt-5 sm:w-24"
              aria-hidden
            />
          </header>

          <div className="mb-10 grid grid-cols-1 gap-8 lg:grid-cols-2">
            <div id="efficiency" className="scroll-mt-32">
              <EfficencyGraph />
            </div>
            <div id="strikes" className="scroll-mt-32">
              <StrikeGraph />
            </div>
          </div>
          
          <div
            id="strike-list"
            className="ui-panel scroll-mt-32 p-6 transition hover:border-emerald-500/30 sm:p-8"
          >
            <StrikeList />
          </div>
        </PageTransition>
      </div>
    </div>
  );
};

export default MainDash;
