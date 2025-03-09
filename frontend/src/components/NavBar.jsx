import { Link } from "react-router-dom";
import logo from "../assets/logo.png";
import { useContext } from "react";
import { AuthContext } from "../MainComponent";
const NavBar = () => {
  const { setAutenticated } = useContext(AuthContext);
  return (
    <nav className="bg-white border-gray-200 dark:bg-[#f8f3d946]">
      <div className="max-w-screen-xl flex flex-wrap items-center justify-between mx-auto p-4">
        <Link
          to="/"
          className="flex items-center space-x-3 rtl:space-x-reverse"
        >
          <img src={logo} className="h-15" alt="Kavach Logo" />
          <span className=" self-center text-3xl uppercase font-bold whitespace-nowrap dark:text-white">
            Kavach
          </span>
        </Link>
        <div className="flex md:order-2 space-x-3 md:space-x-0 rtl:space-x-reverse">
          <button
            type="button"
            className="text-white focus:ring-4 focus:outline-none focus:ring-blue-300 font-medium rounded-lg text-sm px-4 py-2 text-center "
            onClick={() => {
              setAutenticated(false);
            }}
          >
            Log Out
          </button>
        </div>
        <div
          className="items-center justify-between hidden w-full md:flex md:w-auto md:order-1"
          id="navbar-cta"
        >
          <ul className="flex flex-col font-medium p-4 md:p-0 mt-4 border border-gray-100 rounded-lg md:space-x-8 rtl:space-x-reverse md:flex-row md:mt-0 md:border-0 ">
            <li>
              <Link
                to="/"
                className=" block py-2 px-3 md:p-0  rounded-sm "
                aria-current="page"
              >
                Main DashBoard
              </Link>
            </li>
            <li>
              <Link
                to="/SingleSol/0"
                className="block py-2 px-3 md:p-0  rounded-sm "
              >
                Soldier DashBoard
              </Link>
            </li>
            <li>
              <Link to="/tactics" className="block py-2 px-3 md:p-0 rounded-sm">
                Battle Formation
              </Link>
            </li>
          </ul>
        </div>
      </div>
    </nav>
  );
};

export default NavBar;
