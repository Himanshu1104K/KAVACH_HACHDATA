import { useState, useEffect, useContext } from "react";
import { Link, useLocation } from "react-router-dom";
import logo from "../assets/favicon.png";
import { AuthContext } from "../MainComponent";

const NavBar = () => {
  const { setAutenticated } = useContext(AuthContext);
  const [isOpen, setIsOpen] = useState(false);
  const [scrolled, setScrolled] = useState(false);
  const location = useLocation();
  
  // Handle scrolling effects
  useEffect(() => {
    const handleScroll = () => {
      const isScrolled = window.scrollY > 20;
      if (isScrolled !== scrolled) {
        setScrolled(isScrolled);
      }
    };

    window.addEventListener('scroll', handleScroll);
    return () => {
      window.removeEventListener('scroll', handleScroll);
    };
  }, [scrolled]);

  // Close mobile menu when route changes
  useEffect(() => {
    setIsOpen(false);
  }, [location.pathname]);

  // Check if a link is active
  const isActive = (path) => {
    if (path === '/' && location.pathname === '/') return true;
    if (path === '/SingleSol/0' && location.pathname.startsWith('/SingleSol')) return true;
    if (path === '/tactics' && location.pathname === '/tactics') return true;
    return false;
  };

  return (
    <nav className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 ${
      scrolled 
        ? 'bg-black-secondary bg-opacity-90 backdrop-blur-md shadow-lg py-2' 
        : 'bg-black-secondary py-4'
    }`}>
      <div className="max-w-screen-xl flex flex-wrap items-center justify-between mx-auto px-4">
        <Link
          to="/"
          className="flex items-center space-x-3 rtl:space-x-reverse group"
        >
          <img 
            src={logo} 
            className="h-10 group-hover:rotate-12 transition-transform duration-300" 
            alt="Kavach Logo" 
          />
          <span className="self-center text-2xl md:text-3xl uppercase font-bold text-white tracking-widest group-hover:text-gray-accent transition-colors duration-300">
            Kavach
          </span>
        </Link>
        
        {/* Mobile menu button */}
        <button
          type="button"
          className="md:hidden p-2 text-gray-lightest hover:text-white focus:outline-none"
          onClick={() => setIsOpen(!isOpen)}
        >
          <span className="sr-only">Open main menu</span>
          <div className="relative w-6 h-5">
            <span className={`absolute h-0.5 w-6 bg-current transform transition duration-300 ease-in-out ${isOpen ? 'rotate-45 translate-y-2' : '-translate-y-2'}`}></span>
            <span className={`absolute h-0.5 w-6 bg-current transform transition duration-300 ease-in-out ${isOpen ? 'opacity-0' : 'opacity-100'}`}></span>
            <span className={`absolute h-0.5 w-6 bg-current transform transition duration-300 ease-in-out ${isOpen ? '-rotate-45 translate-y-2' : 'translate-y-2'}`}></span>
          </div>
        </button>
        
        <div className="flex md:order-2">
          <button
            type="button"
            className="text-white font-medium rounded-lg text-sm px-5 py-2.5 text-center transition-all duration-300 bg-gradient-to-r from-gray-medium to-gray-dark hover:from-gray-dark hover:to-gray-medium border border-transparent hover:border-gray-light shadow-lg"
            onClick={() => {
              setAutenticated(false);
            }}
          >
            <span className="flex items-center">
              <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1"></path>
              </svg>
              Log Out
            </span>
          </button>
        </div>
        
        {/* Desktop Navigation */}
        <div
          className={`items-center justify-between w-full md:flex md:w-auto md:order-1 ${
            isOpen ? 'block' : 'hidden md:flex'
          }`}
        >
          <ul className="flex flex-col font-medium pt-4 md:pt-0 mt-4 md:mt-0 border-t border-gray-dark md:border-0 md:flex-row md:space-x-1">
            <li>
              <Link
                to="/"
                className={`block py-3 px-4 rounded-lg text-base transition-all duration-300 ${
                  isActive('/') 
                    ? 'bg-gray-dark text-white font-bold' 
                    : 'text-gray-lightest hover:bg-gray-dark hover:text-white'
                }`}
              >
                <span className="flex items-center">
                  <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6"></path>
                  </svg>
                  Main Dashboard
                </span>
              </Link>
            </li>
            <li>
              <Link
                to="/SingleSol/0"
                className={`block py-3 px-4 rounded-lg text-base transition-all duration-300 ${
                  isActive('/SingleSol/0') 
                    ? 'bg-gray-dark text-white font-bold' 
                    : 'text-gray-lightest hover:bg-gray-dark hover:text-white'
                }`}
              >
                <span className="flex items-center">
                  <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z"></path>
                  </svg>
                  Soldier Dashboard
                </span>
              </Link>
            </li>
            <li>
              <Link
                to="/tactics"
                className={`block py-3 px-4 rounded-lg text-base transition-all duration-300 ${
                  isActive('/tactics') 
                    ? 'bg-gray-dark text-white font-bold' 
                    : 'text-gray-lightest hover:bg-gray-dark hover:text-white'
                }`}
              >
                <span className="flex items-center">
                  <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 20l-5.447-2.724A1 1 0 013 16.382V5.618a1 1 0 011.447-.894L9 7m0 13l6-3m-6 3V7m6 10l4.553 2.276A1 1 0 0021 18.382V7.618a1 1 0 00-.553-.894L15 4m0 13V4m0 0L9 7"></path>
                  </svg>
                  Battle Formation
                </span>
              </Link>
            </li>
          </ul>
        </div>
      </div>
    </nav>
  );
};

export default NavBar;
