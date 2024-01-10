
document.getElementById("submitCategories").addEventListener("click",function () {
    document.getElementById("loadingbutton").style.display = "flex"; // show loading info

    const selectedCategories = [];
    document.querySelectorAll(".toggle-button.btn-primary").forEach(bth=> {
        selectedCategories.push(bth.textContent.trim())
    });

    fetch("/recommendations", {
      method:'POST',
      headers: {
        "Content-Type":"application/json",
      },
      body: JSON.stringify({categories:selectedCategories}),
    })
    .then(response => response.json())
    .then(data => {
      const sobeysRecs = data.sobeys.slice(0,12);
      const walmartRecs = data.walmart.slice(0,12);

      document.getElementById("sobeysRecommendations").innerHTML = generateListHTML(sobeysRecs);
      document.getElementById("walmartRecommendations").innerHTML = generateListHTML(walmartRecs);

      document.getElementById("loadingbutton").style.display = "none";
      var recommendationsModal = new bootstrap.Modal(document.getElementById("recommendationsModal"));
      recommendationsModal.show();
    })
    .catch(error => {
        console.error("Error: ",error);
        document.getElementById("loadingbutton").style.display = "none";
    })

    var categoryModal = bootstrap.Modal.getInstance(document.getElementById("categorySelectionModal"));
    categoryModal.hide();
});

  function generateListHTML(items) {
    // Group items by category
    const groupedItems = items.reduce((acc, item) => {
        // Initialize the category key in accumulator
        if (!acc[item.category]) {
            acc[item.category] = [];
        }
        acc[item.category].push(item);
        return acc;
    }, {});

    let table = '<table class="table">';

    // Iterate over each category
    for (const [category, items] of Object.entries(groupedItems)) {
        // Add category as a table header
        table += `<thead>
                    <tr>
                      <th colspan="3">${category}</th>
                    </tr>
                    <tr>
                      <th>Name</th>
                      <th>Price</th>
                      <th>UoM</th>
                    </tr>
                  </thead>`;
        // Add items in this category
        table += '<tbody>';
        items.forEach(item => {
            table += `<tr>
                        <td>${item.name}</td>
                        <td>$${item.price}</td>
                        <td>${item.uom}</td>
                      </tr>`;
        });
        table += '</tbody>';
    }

    table += '</table>';
    return table;
  }