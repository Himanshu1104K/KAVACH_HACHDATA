import { Link } from "react-router-dom";
const CardUpper = (props) => {
  return (
    <>
      <div className="w-[14vw] h-[15vh] p-6  border border-gray-200 rounded-lg shadow-sm bg-[#F8F3D9]">
        <h5 className="mb-2 text-[1.22rem] font-bold text-[#504b3894]">
          {props.title}
        </h5>
        <p className="text-2xl mb-3 font-bold text-[#504b38] inline-block mr-11">
          {props.value}
        </p>
        <button className="cardBtn">
          <p className="inline-flex items-center text-center text-white">
            view
            <svg
              className="rtl:rotate-180 w-3.5 h-3.5 ms-2"
              aria-hidden="true"
              xmlns="http://www.w3.org/2000/svg"
              fill="none"
              viewBox="0 0 14 10"
            >
              <path
                stroke="currentColor"
                stroke-linecap="round"
                stroke-linejoin="round"
                stroke-width="2"
                d="M1 5h12m0 0L9 1m4 4L9 9"
              />
            </svg>
          </p>
        </button>
      </div>
    </>
  );
};

export default CardUpper;
