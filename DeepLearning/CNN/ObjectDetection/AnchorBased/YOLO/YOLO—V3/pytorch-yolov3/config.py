
DATA_WIDTH=416
DATA_HEIGHT=416

CLASS_NUM=3

antors={
    13: [[168,302], [57,221], [336,284]],
    26: [[175,225], [279,160], [249,271]],
    52: [[129,209], [85,413], [44,42]]
}

ANTORS_AREA={
    13: [x*y for x,y in antors[13]],
    26: [x*y for x,y in antors[26]],
    52: [x*y for x,y in antors[52]]
}

