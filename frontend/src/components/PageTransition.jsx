import { useState, useEffect } from 'react';

const PageTransition = ({ children }) => {
  const [animateIn, setAnimateIn] = useState(false);

  useEffect(() => {
    // Trigger animation after component mount
    const timer = setTimeout(() => {
      setAnimateIn(true);
    }, 300);
    return () => clearTimeout(timer);
  }, []);

  return (
    <div className={`transition-opacity duration-1000 ${animateIn ? 'opacity-100' : 'opacity-0'}`}>
      {children}
    </div>
  );
};

export default PageTransition; 