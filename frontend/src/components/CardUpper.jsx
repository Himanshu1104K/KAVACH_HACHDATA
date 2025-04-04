import { Link } from "react-router-dom";
const CardUpper = (props) => {
  return (
    <>
      <div className="w-full h-full p-5 border border-gray-medium rounded-xl shadow-lg bg-gradient-to-br from-black-secondary to-gray-dark hover:from-gray-dark hover:to-black-secondary transition-all duration-300">
        <h5 className="mb-2 text-[1.1rem] font-bold text-gray-lightest tracking-wide">
          {props.title}
        </h5>
        <p className="text-3xl font-bold text-white mb-3">
          {props.value}
        </p>
        <button className="cardBtn mt-1 px-3 py-1 bg-gray-medium hover:bg-gray-light text-white text-xs rounded-md transition-all duration-200 flex items-center opacity-80 hover:opacity-100">
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
