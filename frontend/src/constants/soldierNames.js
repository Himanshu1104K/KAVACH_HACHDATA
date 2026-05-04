export const SOLDIER_NAMES = [
  "Arjun Mehta",
  "Kabir Rana",
  "Vihaan Singh",
  "Ayaan Khanna",
  "Reyansh Kapoor",
  "Ishaan Verma",
  "Advait Nair",
  "Dev Malhotra",
  "Rudra Chauhan",
  "Aryan Bedi",
];

export const getSoldierNameById = (id) => {
  const index = Number(id) - 1;
  return SOLDIER_NAMES[index] || `Soldier ${id}`;
};
