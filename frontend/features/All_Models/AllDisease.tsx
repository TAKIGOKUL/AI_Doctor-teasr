import React from 'react'
import Typography from "@mui/material/Typography";

import { AllDiseases,NewDiseases } from './DiseaseData';
import { Box } from '@mui/material';
import Image from "next/image";
import { useRouter } from "next/router";

function AllDisease() {
  const router = useRouter();

  const handleDiseases = (name: string) => {
    router.push({
      pathname: "/disease/form",
      query: {name}
    },"/disease/form")
  }

  return (
    <>
        <Typography
        // fontFamily={"Poppins"}
        fontStyle={"Bold"}
        fontWeight={"700"}
        fontSize={"96px"}
        lineHeight={"144px"}
        color={"#FFFFFF"}
        sx={{ marginTop: "108px", display: "flex", justifyContent: "center" }}
      >
        Diseases to check 
      </Typography>

      <Box
        sx={{
          display: "flex",
          flexDirection: "row",
          justifyContent: "center",
          alignItems: "center",
          columnGap: "138px",
          mt: "98px",
          cursor:"pointer"
        }}
      >
        {AllDiseases.map((item, index) => {
          return (
            <>
              <Box
                sx={{
                  width: "381px",
                  height:'300px',
                  border: "1px solid #34353A",
                  borderRadius: "18px",
                  background: "#34353A",
                  display: "flex",
                  flexDirection: "column",
                  alignItems: "center",
                  justifyContent:'center'
                }}
                key={index}
                onClick= {() => handleDiseases(item.disease)}
              >
                <Image
                  src={item.image}
                  alt={item.disease}
                  width={100}
                  height={100}
                />
                <Typography
                  fontFamily={"Poppins"}
                  fontStyle={"SemiBold"}
                  fontSize={"42px"}
                  lineHeight={"84px"}
                  color={"#FFFFFF"}
                  sx={{ textAlign: "center" }}
                >
                  {item.disease}
                </Typography>

              </Box>
            </>
          );
        })}
      </Box>

      <br/>


      <Box
        sx={{
          display: "flex",
          flexDirection: "row",
          justifyContent: "center",
          alignItems: "center",
          columnGap: "138px",
          mt: "98px",
          cursor:"pointer"
        }}
      >
        {NewDiseases  .map((item, index) => {
          return (
            <>
              <Box
                sx={{
                  width: "381px",
                  height:'300px',
                  border: "1px solid #34353A",
                  borderRadius: "18px",
                  background: "#34353A",
                  display: "flex",
                  flexDirection: "column",
                  alignItems: "center",
                  justifyContent:'center'
                }}
                key={index}
                onClick= {() => handleDiseases(item.disease)}
              >
                <Image
                  src={item.image}
                  alt={item.disease}
                  width={100}
                  height={100}
                />
                <Typography
                  fontFamily={"Poppins"}
                  fontStyle={"SemiBold"}
                  fontSize={"42px"}
                  lineHeight={"84px"}
                  color={"#FFFFFF"}
                  sx={{ textAlign: "center" }}
                >
                  {item.disease}
                </Typography>

              </Box>
            </>
          );
        })}
      </Box>
    </>
  )
}

export default AllDisease