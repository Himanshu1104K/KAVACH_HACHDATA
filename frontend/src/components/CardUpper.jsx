const CardUpper = (props) => {
  return (
    <>
      <div className="h-full w-full rounded-xl border border-emerald-500/15 bg-[linear-gradient(145deg,rgba(15,23,42,0.92),rgba(11,21,40,0.96))] p-5 shadow-lg transition-all duration-300 hover:border-emerald-400/35 hover:shadow-emerald-950/20">
        <h5 className="mb-2 text-[1.05rem] font-semibold tracking-tight text-slate-200">
          {props.title}
        </h5>
        <p className="mb-3 bg-gradient-to-r from-white to-emerald-100/90 bg-clip-text text-3xl font-bold text-transparent">
          {props.value}
        </p>
        <button className="cardBtn mt-1 flex items-center rounded-full border border-emerald-500/25 bg-emerald-900/50 px-3 py-1.5 text-xs text-emerald-50 transition hover:border-emerald-400/40 hover:bg-emerald-800/60">
          <span className="mr-1">view</span>
          <svg
            className="rtl:rotate-180 w-3 h-3"
            aria-hidden="true"
            xmlns="http://www.w3.org/2000/svg"
            fill="none"
            viewBox="0 0 14 10"
          >
            <path
              stroke="currentColor"
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth="2"
              d="M1 5h12m0 0L9 1m4 4L9 9"
            />
          </svg>
        </button>
      </div>
    </>
  );
};

export default CardUpper;
