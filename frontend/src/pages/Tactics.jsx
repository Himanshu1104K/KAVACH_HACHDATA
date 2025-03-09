import NavBar from "../components/NavBar";
import useFetchTactics from "../customHooks/useFetchTactics";
const Tactics = () => {
  const { formation } = useFetchTactics(
    "https://fastapi-backend-for-kavach-production.up.railway.app/soldier_tacktics"
  );
  return (
    <>
      <NavBar />
      <div className="tactic">
        <h1>{formation?.formation}</h1>
      </div>
    </>
  );
};

export default Tactics;
